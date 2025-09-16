import os
import time
import gc
import json
import psutil
import torch
from collections import defaultdict
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)


class HCPChatbotTrainer:
    """Trainer adapté pour la structure HCP avec corrections pour CUDA multiprocessing."""

    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.hf_token = getattr(config, 'HF_TOKEN', None) or os.getenv('HUGGING_FACE_TOKEN')
        
        # Fix pour CUDA multiprocessing
        if torch.cuda.is_available():
            # Désactiver le multiprocessing pour éviter les erreurs CUDA
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            torch.multiprocessing.set_start_method('spawn', force=True)
        
        self.setup_torch_optimizations()

    def setup_torch_optimizations(self):
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        try:
            print(f"PyTorch threads: {torch.get_num_threads()}")
            print(f"RAM disponible: {psutil.virtual_memory().available / (1024**3):.1f} GB")
            if torch.cuda.is_available():
                print(f"CUDA disponible: {torch.cuda.get_device_name()}")
                print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        except Exception:
            pass

    def initialize_model(self):
        print(f"Initialisation du modèle {self.config.BASE_MODEL}...")

        # Tokenizer
        tokenizer_kwargs = {}
        if self.hf_token:
            tokenizer_kwargs['token'] = self.hf_token

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.BASE_MODEL,
            padding_side="left",
            **tokenizer_kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokens spéciaux depuis la config si disponibles
        required_tokens = []
        try:
            special_map = getattr(self.config, 'HCP_SPECIAL_TOKENS', None)
            if isinstance(special_map, dict):
                required_tokens = list(special_map.values())
        except Exception:
            required_tokens = ["<|hcp|>", "<|territory|>", "<|indicator|>", "<|genre|>", "<|source|>"]

        # Ajouter tokens manquants
        add_tokens = [t for t in required_tokens if t and t not in self.tokenizer.get_vocab()]
        if add_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': add_tokens})
            print(f"Tokens spéciaux ajoutés: {add_tokens}")

        # Chargement modèle avec dtype adapté
        model_kwargs = {}
        if self.hf_token:
            model_kwargs['token'] = self.hf_token

        # Choix dtype
        use_fp16 = bool(getattr(self.config, 'FP16', False)) and torch.cuda.is_available()
        torch_dtype = torch.float16 if use_fp16 else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.BASE_MODEL,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **model_kwargs
        )

        # Resize embeddings si tokens ajoutés
        if add_tokens:
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Déplacer sur device si cuda
        if torch.cuda.is_available():
            try:
                self.model.to('cuda')
                print("Modèle déplacé sur CUDA")
            except Exception as e:
                print(f"⚠️ Impossible de déplacer le modèle sur CUDA: {e}")

        print(f"Modèle initialisé: {self.config.BASE_MODEL}")

    def format_dialogue_for_training(self, data_item):
        """Formate les données pour l'entraînement avec contexte HCP enrichi."""
        if 'conversation' in data_item and data_item['conversation']:
            return data_item['conversation']

        question = data_item.get('input_text', data_item.get('question', '')).strip()
        answer = data_item.get('target_text', data_item.get('answer', '')).strip()

        territory = data_item.get('territory', data_item.get('original_territory', ''))
        indicator = data_item.get('variable', data_item.get('indicateur', ''))
        genre = data_item.get('sexe', data_item.get('genre', ''))
        source = data_item.get('source_data', data_item.get('source', ''))

        context_parts = ['<|hcp|>']
        if territory and territory not in ['Territoire non spécifié', '']:
            context_parts.append(f"<|territory|>{territory}")
        if indicator and indicator not in ['', 'indicateur_demographique']:
            context_parts.append(f"<|indicator|>{indicator}")
        if genre and genre not in ['', 'non spécifié']:
            context_parts.append(f"<|genre|>{genre}")
        if source and source not in ['', 'unknown']:
            context_parts.append(f"<|source|>{source}")

        context_prefix = ''.join(context_parts)
        formatted = f"{context_prefix}<|user|>{question}<|assistant|>{answer}<|endoftext|>"
        return formatted

    def tokenize_data(self, examples):
        """Tokenisation sans multiprocessing pour éviter les erreurs CUDA."""
        conversations = []
        length = len(examples['input_text'])
        for i in range(length):
            data_item = {
                'input_text': examples['input_text'][i],
                'target_text': examples['target_text'][i],
                'territory': (examples.get('territory') or [''])[i] if 'territory' in examples else '',
                'variable': (examples.get('variable') or [''])[i] if 'variable' in examples else '',
                'sexe': (examples.get('sexe') or [''])[i] if 'sexe' in examples else '',
                'source_data': (examples.get('source_data') or [''])[i] if 'source_data' in examples else ''
            }
            conversations.append(self.format_dialogue_for_training(data_item))

        inputs = self.tokenizer(
            conversations,
            truncation=True,
            padding='max_length',
            max_length=self.config.MAX_LENGTH,
            return_tensors=None
        )

        inputs['labels'] = inputs['input_ids'].copy()
        return inputs

    def prepare_dataset(self, training_data):
        """Préparation du dataset sans multiprocessing."""
        print(f"Préparation du dataset avec {len(training_data)} échantillons...")
        cleaned_data = []

        field_mapping = {
            'input_text': ['input_text', 'question'],
            'target_text': ['target_text', 'answer'],
            'territory': ['territory', 'original_territory', 'territoire'],
            'variable': ['variable', 'indicator', 'indicateur'],
            'sexe': ['sexe', 'genre'],
            'source_data': ['source_data', 'source'],
            'question_type': ['question_type'],
            'data_source': ['data_source']
        }

        for item in training_data:
            cleaned_item = {}
            for target_field, source_fields in field_mapping.items():
                val = None
                for sf in source_fields:
                    if sf in item and item[sf]:
                        val = item[sf]
                        break
                cleaned_item[target_field] = '' if val is None else str(val).strip()

            if cleaned_item['input_text'] and cleaned_item['target_text']:
                if cleaned_item.get('sexe') in ['', 'non spécifié']:
                    cleaned_item['sexe'] = 'ensemble'
                if cleaned_item.get('territory') == '':
                    cleaned_item['territory'] = 'Territoire non spécifié'
                cleaned_data.append(cleaned_item)

        if not cleaned_data:
            raise ValueError("Aucune donnée valide après nettoyage")

        dataset = Dataset.from_list(cleaned_data)

        # Tokenisation sans multiprocessing
        tokenized = dataset.map(
            self.tokenize_data,
            batched=True,
            num_proc=None,
            remove_columns=dataset.column_names,
            desc="Tokenisation"
        )

        return tokenized

    def create_stratified_split(self, training_data, train_ratio=0.7, val_ratio=0.3):
        """
        Crée un split stratifié 70% train / 30% validation basé sur:
        - Territory (pour couvrir tous les territoires)
        - Indicator type (pour couvrir tous les types d'indicateurs)
        - Genre (pour équilibrer masculin/féminin/ensemble)
        """
        import random
        
        print(f"Création du split stratifié: {train_ratio:.0%} train / {val_ratio:.0%} validation")
        
        # Grouper par (territoire, indicateur, genre) pour stratification
        strata = defaultdict(list)
        for i, item in enumerate(training_data):
            territory = item.get('territory', 'Unknown')
            indicator = item.get('variable', 'unknown')
            genre = item.get('sexe', 'ensemble')
            strata_key = f"{territory}_{indicator}_{genre}"
            strata[strata_key].append(i)
        
        print(f"Nombre de strates identifiées: {len(strata)}")
        
        train_indices = []
        val_indices = []
        
        for strata_key, indices in strata.items():
            random.shuffle(indices)
            # CORRECTION: calculer train en premier
            n_train = max(1, int(len(indices) * train_ratio))
            n_val = len(indices) - n_train
            
            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:])
        
        # Mélanger les indices finaux
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        
        train_data = [training_data[i] for i in train_indices]
        val_data = [training_data[i] for i in val_indices]
        
        print(f"Split créé:")
        print(f"  - Entraînement: {len(train_data)} échantillons ({len(train_data)/len(training_data)*100:.1f}%)")
        print(f"  - Validation: {len(val_data)} échantillons ({len(val_data)/len(training_data)*100:.1f}%)")
        
        # Vérification de la distribution
        self._verify_split_distribution(train_data, val_data)
        
        return train_data, val_data
        
    def _verify_split_distribution(self, train_data, val_data):
        """Vérifie la distribution des données dans le split."""
        
        def get_distribution(data, key):
            dist = defaultdict(int)
            for item in data:
                dist[item.get(key, 'Unknown')] += 1
            return dist
        
        print("\n📊 Vérification de la distribution:")
        
        # Distribution par genre
        train_genre = get_distribution(train_data, 'sexe')
        val_genre = get_distribution(val_data, 'sexe')
        print(f"Genre - Train: {dict(train_genre)}")
        print(f"Genre - Val: {dict(val_genre)}")
        
        # Distribution par source
        train_source = get_distribution(train_data, 'source_data')
        val_source = get_distribution(val_data, 'source_data')
        print(f"Sources - Train: {len(train_source)} types")
        print(f"Sources - Val: {len(val_source)} types")
        
        # Territoires uniques
        train_territories = set(item.get('territory', 'Unknown') for item in train_data)
        val_territories = set(item.get('territory', 'Unknown') for item in val_data)
        common_territories = train_territories.intersection(val_territories)
        
        print(f"Territoires - Train: {len(train_territories)} uniques")
        print(f"Territoires - Val: {len(val_territories)} uniques")
        print(f"Territoires communs: {len(common_territories)} ({len(common_territories)/len(train_territories)*100:.1f}%)")

    def calculate_compatible_steps(self, dataset_size, batch_size, num_epochs, use_validation=False):
        grad_acum = getattr(self.config, 'GRADIENT_ACCUMULATION_STEPS', 1)
        steps_per_epoch = max(1, dataset_size // (batch_size * grad_acum))
        if dataset_size % (batch_size * grad_acum) != 0:
            steps_per_epoch += 1
        total_steps = steps_per_epoch * num_epochs
        
        logging_steps = max(5, min(steps_per_epoch // 10, 50))
        
        if use_validation:
            eval_steps = max(10, steps_per_epoch // 3)
            # S'assurer que save_steps est un multiple de eval_steps
            save_steps = eval_steps
        else:
            eval_steps = None
            save_steps = min(steps_per_epoch // 2, 500)

        return {
            'total_steps': total_steps,
            'steps_per_epoch': steps_per_epoch,
            'logging_steps': logging_steps,
            'save_steps': save_steps,
            'eval_steps': eval_steps
        }

    def _validate_hcp_training_data(self, training_data):
        if not training_data:
            raise ValueError("Aucune donnée fournie")
        
        valid = 0
        for item in training_data[:200]:
            if isinstance(item, dict) and (item.get('question') or item.get('input_text')) and (item.get('answer') or item.get('target_text')):
                valid += 1
        
        if valid == 0:
            raise ValueError("Aucun item valide détecté dans les 200 premiers éléments")
        
        print(f"✅ Validation réussie: {valid}/200 échantillons valides testés")
        return True

    def save_hcp_enhanced_metadata(self, train_data, training_time, val_data=None, original_size=None):
        """Sauvegarde des métadonnées d'entraînement enrichies."""
        metadata = {
            'training_time_minutes': training_time,
            'original_dataset_size': original_size or len(train_data),
            'train_size': len(train_data),
            'validation_size': len(val_data) if val_data else 0,
            'model_config': {
                'base_model': self.config.BASE_MODEL,
                'max_length': self.config.MAX_LENGTH,
                'batch_size': self.config.BATCH_SIZE,
                'learning_rate': self.config.LEARNING_RATE,
                'num_epochs': self.config.NUM_EPOCHS,
                'fp16': getattr(self.config, 'FP16', False)
            },
            'data_distribution': {
                'territories_count': len(set(item.get('territory', 'Unknown') for item in train_data)),
                'indicators_count': len(set(item.get('variable', 'unknown') for item in train_data)),
                'sources_count': len(set(item.get('source_data', 'unknown') for item in train_data))
            }
        }
        
        os.makedirs(self.config.MODEL_PATH, exist_ok=True)
        path = os.path.join(self.config.MODEL_PATH, 'hcp_enhanced_metadata.json')
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Métadonnées sauvées: {path}")
        return metadata

    def print_hcp_training_summary(self, training_data, training_time, original_size, val_data=None):
        print(f"\n🎯 RÉSUMÉ D'ENTRAÎNEMENT HCP")
        print(f"{'='*50}")
        print(f"Temps d'entraînement: {training_time:.1f} minutes")
        print(f"Dataset original: {original_size:,} échantillons")
        print(f"Entraînement: {len(training_data):,} échantillons")
        if val_data:
            print(f"Validation: {len(val_data):,} échantillons")
        print(f"Modèle sauvé: {self.config.MODEL_PATH}")
        print(f"{'='*50}")

    def train_model(self, training_data, train_ratio=0.7, val_ratio=0.3):
        """
        Entraîne le modèle avec un split train/validation automatique.
        
        Args:
            training_data: Liste des données d'entraînement
            train_ratio: Ratio pour l'entraînement (défaut: 0.7)
            val_ratio: Ratio pour la validation (défaut: 0.3)
        """
        start_time = time.time()
        original_size = len(training_data)
        
        print(f"🚀 Début de l'entraînement HCP avec {original_size:,} échantillons")
        
        if self.model is None:
            self.initialize_model()

        self._validate_hcp_training_data(training_data)

        # Créer le split stratifié
        train_data, val_data = self.create_stratified_split(
            training_data, 
            train_ratio=train_ratio, 
            val_ratio=val_ratio
        )

        # Préparer les datasets
        print("\n🔄 Préparation des datasets...")
        train_dataset = self.prepare_dataset(train_data)
        val_dataset = self.prepare_dataset(val_data) if val_data else None

        # Configuration des étapes d'entraînement
        steps_cfg = self.calculate_compatible_steps(
            len(train_dataset), 
            self.config.BATCH_SIZE, 
            self.config.NUM_EPOCHS, 
            use_validation=bool(val_dataset)
        )

        print(f"\n⚙️ Configuration d'entraînement:")
        print(f"  - Steps par époque: {steps_cfg['steps_per_epoch']}")
        print(f"  - Steps total: {steps_cfg['total_steps']}")
        print(f"  - Logging steps: {steps_cfg['logging_steps']}")
        print(f"  - Save steps: {steps_cfg['save_steps']}")
        if steps_cfg['eval_steps']:
            print(f"  - Eval steps: {steps_cfg['eval_steps']}")

        # Arguments d'entraînement
        training_args = TrainingArguments(
            output_dir=self.config.MODEL_PATH,
            overwrite_output_dir=True,
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=getattr(self.config, 'GRADIENT_ACCUMULATION_STEPS', 1),
            learning_rate=self.config.LEARNING_RATE,
            weight_decay=getattr(self.config, 'WEIGHT_DECAY', 0.01),
            warmup_steps=max(100, steps_cfg['total_steps'] // 20),
            dataloader_num_workers=0,
            remove_unused_columns=False,
            logging_steps=steps_cfg['logging_steps'],
            save_steps=steps_cfg['save_steps'],
            save_total_limit=3,
            evaluation_strategy="steps" if steps_cfg['eval_steps'] else "no",
            eval_steps=steps_cfg['eval_steps'],
            fp16=bool(getattr(self.config, 'FP16', False)) and torch.cuda.is_available(),
            report_to=None,
            load_best_model_at_end=False,  # Désactivé pour éviter les problèmes
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            seed=42
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )

        try:
            print(f"\n🎯 Démarrage de l'entraînement...")
            trainer.train()
            
            print(f"\n💾 Sauvegarde du modèle...")
            trainer.save_model()
            self.tokenizer.save_pretrained(self.config.MODEL_PATH)
            
            elapsed_time = (time.time() - start_time) / 60
            
            # Sauvegarder les métadonnées
            self.save_hcp_enhanced_metadata(train_data, elapsed_time, val_data, original_size)
            
            # Afficher le résumé
            self.print_hcp_training_summary(train_data, elapsed_time, original_size, val_data)
            
            print(f"✅ Entraînement terminé avec succès!")
            
        except RuntimeError as e:
            print(f"❌ Erreur d'entraînement (RuntimeError): {e}")
            if 'out of memory' in str(e).lower():
                print("💡 Suggestions pour résoudre l'erreur OOM:")
                print("  - Réduire BATCH_SIZE dans config.py")
                print("  - Réduire MAX_LENGTH dans config.py")
                print("  - Activer gradient_checkpointing (déjà activé)")
                print("  - Utiliser un modèle plus petit")
            raise
        except Exception as e:
            print(f"❌ Erreur inattendue: {e}")
            raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_model_info(self):
        """Retourne des informations sur le modèle entraîné."""
        if self.model is None:
            return {"status": "Model not initialized"}
        
        return {
            "model_name": self.config.BASE_MODEL,
            "vocab_size": len(self.tokenizer),
            "special_tokens": len(self.tokenizer.additional_special_tokens),
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "model_size_mb": sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024*1024),
            "device": next(self.model.parameters()).device if self.model else "unknown"
        }


# Fonction utilitaire pour l'entraînement facile
def train_hcp_model_with_split(config, training_data, train_ratio=0.7, val_ratio=0.3):
    """
    Fonction helper pour entraîner facilement un modèle HCP avec split automatique.
    
    Args:
        config: Configuration HCP
        training_data: Données d'entraînement
        train_ratio: Ratio d'entraînement (défaut: 70%)
        val_ratio: Ratio de validation (défaut: 30%)
    
    Returns:
        HCPChatbotTrainer: Instance du trainer avec modèle entraîné
    """
    trainer = HCPChatbotTrainer(config)
    trainer.train_model(training_data, train_ratio=train_ratio, val_ratio=val_ratio)
    
    print(f"\n📊 Informations sur le modèle entraîné:")
    model_info = trainer.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    return trainer


def check_system_resources_hcp(min_ram_gb: float = 8.0):
    """Vérifie les ressources système pour l'entraînement HCP."""
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    
    available_gb = memory.available / (1024**3)
    ready = available_gb >= min_ram_gb
    
    cuda_info = {}
    if torch.cuda.is_available():
        cuda_info = {
            "cuda_available": True,
            "device_name": torch.cuda.get_device_name(),
            "cuda_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "cuda_memory_allocated": torch.cuda.memory_allocated() / (1024**3),
            "cuda_memory_cached": torch.cuda.memory_reserved() / (1024**3)
        }
    else:
        cuda_info = {"cuda_available": False}
    
    return {
        'ready': ready,
        'memory_available_gb': available_gb,
        'memory_total_gb': memory.total / (1024**3),
        'cpu_usage_percent': cpu,
        'cpu_count': os.cpu_count(),
        **cuda_info
    }


# Script de test
if __name__ == "__main__":
    print("🧪 Test du trainer HCP corrigé\n")
    
    # Vérifier les ressources
    resources = check_system_resources_hcp()
    print(f"💻 Ressources système:")
    for key, value in resources.items():
        print(f"  {key}: {value}")
    
    if not resources['ready']:
        print(f"⚠️ RAM insuffisante (besoin: 8GB, disponible: {resources['memory_available_gb']:.1f}GB)")
    else:
        print(f"✅ Ressources suffisantes pour l'entraînement")
