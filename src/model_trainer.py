# import torch
# from transformers import (
#     AutoTokenizer, AutoModelForCausalLM, 
#     TrainingArguments, Trainer, 
#     DataCollatorForLanguageModeling
# )
# from datasets import Dataset
# import json
# import os
# import time
# import psutil
# import gc

# class HCPChatbotTrainer:
#     def __init__(self, config):
#         self.config = config
#         self.tokenizer = None
#         self.model = None
#         self.hf_token = getattr(config, 'HF_TOKEN', None) or os.getenv('HUGGING_FACE_TOKEN')
#         self.setup_torch_optimizations()

#     def setup_torch_optimizations(self):
#         """Configuration optimale pour PyTorch"""
#         torch.set_num_threads(8)
#         if not torch.cuda.is_available():
#             try:
#                 torch.backends.mkl.enabled = True
#             except:
#                 pass
        
#         print(f"Threads PyTorch: {torch.get_num_threads()}")
#         print(f"RAM disponible: {psutil.virtual_memory().available / (1024**3):.1f} GB")

#     def initialize_model(self):
#         """Initialisation du modèle et tokenizer avec support nouvelle structure"""
#         print(f"Initialisation du modèle {self.config.BASE_MODEL}...")
        
#         # Tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.config.BASE_MODEL,
#             padding_side="left",
#             token=self.hf_token
#         )
        
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
        
#         # NOUVEAU: Tokens spéciaux adaptés à la nouvelle structure
#         special_tokens = []
#         required_tokens = [
#             "<|user|>", "<|assistant|>", 
#             "<|territory|>", "<|context|>", 
#             "<|indicator|>", "<|genre|>"  # Nouveaux tokens pour la structure
#         ]
        
#         for token in required_tokens:
#             if token not in self.tokenizer.get_vocab():
#                 special_tokens.append(token)
        
#         if special_tokens:
#             self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
#             print(f"Tokens spéciaux ajoutés: {special_tokens}")

#         # Modèle
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.config.BASE_MODEL,
#             torch_dtype=torch.float32,
#             low_cpu_mem_usage=True,
#             token=self.hf_token
#         )
        
#         if special_tokens:
#             self.model.resize_token_embeddings(len(self.tokenizer))
#             print("Embeddings redimensionnés pour les nouveaux tokens")
        
#         print(f"Modèle {self.config.BASE_MODEL} initialisé avec succès")
#         print(f"Paramètres du modèle: {self.model.num_parameters():,}")

#     def format_dialogue_for_training(self, data_item):
#         """Formate les dialogues pour l'entraînement avec contexte enrichi"""
        
#         # Si c'est déjà une conversation formatée
#         if 'conversation' in data_item and data_item['conversation']:
#             return data_item['conversation']
        
#         # NOUVEAU: Formatage enrichi avec métadonnées de la nouvelle structure
#         question = data_item.get('input_text', data_item.get('question', ''))
#         answer = data_item.get('target_text', data_item.get('answer', ''))
        
#         # Contexte enrichi basé sur la nouvelle structure
#         territory = data_item.get('territory', data_item.get('original_territory', ''))
#         indicator = data_item.get('variable', data_item.get('indicator', ''))
#         genre = data_item.get('sexe', data_item.get('genre', ''))
        
#         # Format conditionnel selon les métadonnées disponibles
#         formatted_parts = []
        
#         if territory and territory != 'Territoire non spécifié':
#             formatted_parts.append(f"<|territory|>{territory}")
        
#         if indicator and indicator != 'indicateur démographique':
#             formatted_parts.append(f"<|indicator|>{indicator}")
            
#         if genre and genre not in ['non spécifié', '']:
#             formatted_parts.append(f"<|genre|>{genre}")
        
#         # Construction du dialogue final
#         context_prefix = "".join(formatted_parts)
#         formatted = f"{context_prefix}<|user|>{question}<|assistant|>{answer}<|endoftext|>"
        
#         return formatted

#     def tokenize_data(self, examples):
#         """Tokenisation des données avec support du nouveau format"""
#         conversations = []
        
#         # NOUVEAU: Gérer les deux formats de données
#         for i in range(len(examples['input_text'])):
#             data_item = {
#                 'input_text': examples['input_text'][i],
#                 'target_text': examples['target_text'][i],
#                 'territory': examples.get('territory', [''])[i] if 'territory' in examples else '',
#                 'variable': examples.get('variable', [''])[i] if 'variable' in examples else '',
#                 'sexe': examples.get('sexe', [''])[i] if 'sexe' in examples else '',
#             }
            
#             formatted_conversation = self.format_dialogue_for_training(data_item)
#             conversations.append(formatted_conversation)
        
#         # Tokenisation
#         inputs = self.tokenizer(
#             conversations,
#             truncation=True,
#             padding='max_length',
#             max_length=self.config.MAX_LENGTH,
#             return_tensors=None
#         )
        
#         inputs["labels"] = inputs["input_ids"].copy()
#         return inputs

#     def prepare_dataset(self, training_data):
#         """Préparation du dataset avec validation renforcée"""
#         print(f"Préparation du dataset avec {len(training_data)} échantillons...")
        
#         # NOUVEAU: Nettoyage spécifique à la nouvelle structure
#         cleaned_data = []
#         for item in training_data:
#             cleaned_item = {}
            
#             # Mapping des champs standards
#             field_mapping = {
#                 'input_text': ['input_text', 'question'],
#                 'target_text': ['target_text', 'answer'],
#                 'territory': ['territory', 'original_territory'],
#                 'variable': ['variable', 'indicator', 'indicateur'],
#                 'sexe': ['sexe', 'genre'],
#                 'question_type': ['question_type'],
#                 'data_source': ['data_source']
#             }
            
#             for target_field, source_fields in field_mapping.items():
#                 value = None
#                 for source_field in source_fields:
#                     if source_field in item and item[source_field]:
#                         value = item[source_field]
#                         break
                
#                 # Nettoyage des valeurs
#                 if isinstance(value, (list, dict, tuple)):
#                     cleaned_item[target_field] = str(value) if value else ""
#                 elif value is None:
#                     cleaned_item[target_field] = ""
#                 else:
#                     cleaned_item[target_field] = str(value).strip()
            
#             # Validation des champs obligatoires
#             if cleaned_item.get('input_text') and cleaned_item.get('target_text'):
#                 cleaned_data.append(cleaned_item)
        
#         print(f"  ✓ {len(cleaned_data)} échantillons valides après nettoyage")
        
#         if not cleaned_data:
#             raise ValueError("Aucune donnée valide après nettoyage!")
        
#         # Création du dataset
#         dataset = Dataset.from_list(cleaned_data)
        
#         # Tokenisation
#         tokenized_dataset = dataset.map(
#             self.tokenize_data,
#             batched=True,
#             num_proc=min(4, len(cleaned_data) // 100 + 1),
#             remove_columns=dataset.column_names,
#             desc="Tokenisation des données avec nouveau format"
#         )
        
#         print(f"Dataset tokenisé: {len(tokenized_dataset)} échantillons")
#         return tokenized_dataset

#     def create_training_subset(self, training_data, max_samples=5000, balanced=True):
#         """Création d'un sous-ensemble équilibré pour l'entraînement"""
#         if len(training_data) <= max_samples:
#             return training_data
        
#         print(f"Création d'un sous-ensemble de {max_samples} échantillons à partir de {len(training_data)}")
        
#         if not balanced:
#             import random
#             random.shuffle(training_data)
#             return training_data[:max_samples]
        
#         # NOUVEAU: Équilibrage basé sur la nouvelle structure
#         grouping_keys = ['territory', 'question_type', 'sexe']
#         groups = {}
        
#         for item in training_data:
#             # Créer une clé de groupe basée sur les métadonnées disponibles
#             group_key_parts = []
#             for key in grouping_keys:
#                 value = item.get(key, 'unknown')
#                 if value and value not in ['non spécifié', 'Territoire non spécifié', '']:
#                     group_key_parts.append(f"{key}:{value}")
            
#             group_key = "_".join(group_key_parts) if group_key_parts else "general"
            
#             if group_key not in groups:
#                 groups[group_key] = []
#             groups[group_key].append(item)
        
#         # Distribution équilibrée
#         samples_per_group = max(1, max_samples // len(groups))
#         remaining_samples = max_samples
#         subset = []
        
#         for group_items in groups.values():
#             if remaining_samples <= 0:
#                 break
#             take = min(samples_per_group, len(group_items), remaining_samples)
#             subset.extend(group_items[:take])
#             remaining_samples -= take
        
#         # Statistiques du sous-ensemble
#         subset_stats = {
#             'territories': set(),
#             'question_types': {},
#             'data_sources': {}
#         }
        
#         for item in subset:
#             subset_stats['territories'].add(item.get('territory', 'Unknown'))
            
#             q_type = item.get('question_type', 'unknown')
#             subset_stats['question_types'][q_type] = subset_stats['question_types'].get(q_type, 0) + 1
            
#             source = item.get('data_source', 'unknown')
#             subset_stats['data_sources'][source] = subset_stats['data_sources'].get(source, 0) + 1
        
#         print(f"  Sous-ensemble créé: {len(subset)} échantillons")
#         print(f"  {len(subset_stats['territories'])} territoires représentés")
#         print(f"  {len(subset_stats['question_types'])} types de questions")
#         print(f"  Sources: {dict(list(subset_stats['data_sources'].items()))}")
        
#         return subset

#     def estimate_training_time(self, num_samples):
#         """Estimation du temps d'entraînement"""
#         # Facteurs d'ajustement
#         base_samples_per_second = 1.5
        
#         if self.config.MAX_LENGTH > 512:
#             base_samples_per_second *= 0.8
#         if self.config.BATCH_SIZE > 2:
#             base_samples_per_second *= 1.2
#         if hasattr(self.config, 'NEW_DATA_STRUCTURE'):
#             # La nouvelle structure peut être légèrement plus lente à traiter
#             base_samples_per_second *= 0.9
        
#         total_samples = num_samples * self.config.NUM_EPOCHS
#         estimated_time = total_samples / base_samples_per_second / 60
        
#         print(f"Temps d'entraînement estimé: {estimated_time:.1f} minutes ({estimated_time/60:.1f} heures)")
        
#         recommend_subset = False
#         if estimated_time > 120:
#             print("⚠️ Attention: L'entraînement pourrait être long!")
#             print("Recommandations d'optimisation:")
#             print("- Réduire NUM_EPOCHS (2-3 epochs suffisent souvent)")
#             print("- Réduire MAX_LENGTH si vos dialogues sont courts")
#             print("- Augmenter BATCH_SIZE si la RAM le permet")
#             print("- Utiliser create_training_subset() pour un test rapide")
            
#             if estimated_time > 360:
#                 print("🚨 RECOMMANDATION FORTE: Test avec un sous-ensemble d'abord!")
#                 recommend_subset = True
        
#         return estimated_time, recommend_subset

#     def create_validation_split(self, training_data, val_ratio=0.1):
#         """Création du split validation avec équilibrage territorial"""
#         if val_ratio <= 0:
#             return training_data, []
        
#         # NOUVEAU: Split équilibré par territoire ET type de question
#         territory_type_groups = {}
        
#         for i, item in enumerate(training_data):
#             territory = item.get('territory', 'Unknown')
#             q_type = item.get('question_type', 'unknown')
#             key = f"{territory}_{q_type}"
            
#             if key not in territory_type_groups:
#                 territory_type_groups[key] = []
#             territory_type_groups[key].append(i)
        
#         train_indices = []
#         val_indices = []
        
#         for indices in territory_type_groups.values():
#             val_size = max(1, int(len(indices) * val_ratio))
#             val_indices.extend(indices[:val_size])
#             train_indices.extend(indices[val_size:])
        
#         train_data = [training_data[i] for i in train_indices]
#         val_data = [training_data[i] for i in val_indices]
        
#         print(f"Split validation équilibré: {len(train_data)} train, {len(val_data)} validation")
#         return train_data, val_data

#     def calculate_compatible_steps(self, dataset_size, batch_size, num_epochs, use_validation=False):
#         """Calcul des steps compatibles pour éviter les erreurs"""
#         gradient_accumulation_steps = getattr(self.config, 'GRADIENT_ACCUMULATION_STEPS', 1)
        
#         # Steps par époque
#         steps_per_epoch = dataset_size // (batch_size * gradient_accumulation_steps)
#         if dataset_size % (batch_size * gradient_accumulation_steps) != 0:
#             steps_per_epoch += 1
        
#         # Total steps
#         total_steps = steps_per_epoch * num_epochs
        
#         if use_validation:
#             # Eval steps compatible
#             target_eval_ratio = 0.2
#             target_eval_steps = max(50, int(steps_per_epoch * target_eval_ratio))
            
#             eval_steps = target_eval_steps
#             while steps_per_epoch % eval_steps != 0 and eval_steps > 10:
#                 eval_steps -= 1
            
#             if eval_steps <= 10:
#                 eval_steps = steps_per_epoch
            
#             save_steps = eval_steps
#         else:
#             eval_steps = None
#             save_steps = min(steps_per_epoch, 500)
        
#         logging_steps = max(10, min(steps_per_epoch // 10, 100))
        
#         print(f"Configuration des steps pour nouvelle structure:")
#         print(f"   - Dataset: {dataset_size} échantillons")
#         print(f"   - Batch effectif: {batch_size * gradient_accumulation_steps}")
#         print(f"   - Steps/époque: {steps_per_epoch}")
#         print(f"   - Total steps: {total_steps}")
        
#         return {
#             'total_steps': total_steps,
#             'steps_per_epoch': steps_per_epoch,
#             'logging_steps': logging_steps,
#             'save_steps': save_steps,
#             'eval_steps': eval_steps
#         }

#     def train_model(self, training_data, use_validation=False, use_subset=False, max_subset_size=5000):
#         """Entraînement principal avec support de la nouvelle structure"""
#         start_time = time.time()
        
#         if self.model is None:
#             self.initialize_model()
        
#         original_size = len(training_data)
        
#         # NOUVEAU: Validation de la structure des données
#         self._validate_training_data_structure(training_data)
        
#         # Estimation et recommandations
#         estimated_time, recommend_subset = self.estimate_training_time(len(training_data))
        
#         if use_subset or (recommend_subset and len(training_data) > max_subset_size):
#             if not use_subset and recommend_subset:
#                 print(f"\n💡 RECOMMANDATION: Test avec {max_subset_size} échantillons")
#                 response = input("Continuer avec un sous-ensemble ? (y/N): ").lower().strip()
#                 if response in ['y', 'yes', 'oui', 'o']:
#                     use_subset = True
            
#             if use_subset:
#                 training_data = self.create_training_subset(training_data, max_subset_size, balanced=True)
#                 print(f"✅ Sous-ensemble: {len(training_data)} échantillons (vs {original_size} originaux)")
        
#         # Split validation
#         if use_validation:
#             train_data, val_data = self.create_validation_split(training_data, 0.1)
#         else:
#             train_data = training_data
#             val_data = []
        
#         # Préparation des datasets
#         train_dataset = self.prepare_dataset(train_data)
#         val_dataset = self.prepare_dataset(val_data) if val_data else None
        
#         # Création du répertoire modèle
#         os.makedirs(self.config.MODEL_PATH, exist_ok=True)
        
#         # Configuration des steps
#         steps_config = self.calculate_compatible_steps(
#             len(train_dataset), 
#             self.config.BATCH_SIZE, 
#             self.config.NUM_EPOCHS,
#             use_validation and val_dataset is not None
#         )
        
#         # Arguments d'entraînement
#         training_args = TrainingArguments(
#             output_dir=self.config.MODEL_PATH,
#             overwrite_output_dir=True,
            
#             # Époques et steps
#             num_train_epochs=self.config.NUM_EPOCHS,
#             max_steps=-1,
            
#             # Batch et apprentissage
#             per_device_train_batch_size=self.config.BATCH_SIZE,
#             gradient_accumulation_steps=getattr(self.config, 'GRADIENT_ACCUMULATION_STEPS', 1),
#             learning_rate=self.config.LEARNING_RATE,
#             weight_decay=getattr(self.config, 'WEIGHT_DECAY', 0.01),
            
#             # Dataloaders
#             dataloader_num_workers=min(4, getattr(self.config, 'DATALOADER_NUM_WORKERS', 2)),
#             dataloader_pin_memory=False,
#             remove_unused_columns=False,
            
#             # Logging et sauvegarde
#             logging_steps=steps_config['logging_steps'],
#             save_steps=steps_config['save_steps'],
#             save_total_limit=2,
            
#             # Évaluation
#             evaluation_strategy="steps" if steps_config['eval_steps'] else "no",
#             eval_steps=steps_config['eval_steps'],
            
#             # Optimisations
#             prediction_loss_only=True,
#             report_to=None,
#             load_best_model_at_end=bool(steps_config['eval_steps']),
#             metric_for_best_model="eval_loss" if steps_config['eval_steps'] else None,
#             gradient_checkpointing=True,
#             bf16=False,
#             fp16=False,
#         )
        
#         # Data collator
#         data_collator = DataCollatorForLanguageModeling(
#             tokenizer=self.tokenizer,
#             mlm=False,
#         )
        
#         # Trainer
#         trainer = Trainer(
#             model=self.model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=val_dataset,
#             data_collator=data_collator,
#         )
        
#         # Entraînement
#         print(f"\n🚀 DÉBUT DE L'ENTRAÎNEMENT")
#         print(f"Structure: {'Nouvelle (qa_pairs)' if any(item.get('data_source') == 'nouvelle_structure' for item in training_data) else 'Legacy'}")
#         print(f"Configuration: {self.config.NUM_EPOCHS} époques exactes")
        
#         try:
#             memory_before = psutil.virtual_memory()
#             print(f"RAM avant: {memory_before.percent:.1f}%")
            
#             gc.collect()
#             trainer.train()
            
#             # Sauvegarde
#             print("💾 Sauvegarde du modèle...")
#             trainer.save_model()
#             self.tokenizer.save_pretrained(self.config.MODEL_PATH)
            
#             training_time = (time.time() - start_time) / 60
            
#             # Métadonnées enrichies
#             self.save_enhanced_metadata(training_data, training_time, val_data, original_size)
            
#             print(f"\n✅ ENTRAÎNEMENT TERMINÉ EN {training_time:.1f} minutes!")
#             print(f"📁 Modèle sauvé: {self.config.MODEL_PATH}")
            
#             self.print_enhanced_training_summary(training_data, training_time, original_size)
            
#         except Exception as e:
#             print(f"❌ Erreur pendant l'entraînement: {e}")
#             raise
#         finally:
#             gc.collect()

#     def _validate_training_data_structure(self, training_data):
#         """Valide la structure des données d'entraînement"""
#         if not training_data:
#             raise ValueError("Aucune donnée d'entraînement fournie!")
        
#         print("🔍 Validation de la structure des données...")
        
#         required_fields = ['question', 'answer']
#         optional_fields = ['territory', 'question_type', 'variable', 'sexe', 'data_source']
        
#         issues = []
#         structure_stats = {
#             'total_items': len(training_data),
#             'valid_items': 0,
#             'missing_fields': {},
#             'data_sources': {},
#             'has_new_structure': False,
#             'has_legacy_structure': False
#         }
        
#         for i, item in enumerate(training_data[:100]):  # Vérifier les 100 premiers
#             if not isinstance(item, dict):
#                 issues.append(f"Item {i}: n'est pas un dictionnaire")
#                 continue
            
#             # Vérifier les champs requis
#             has_question = any(field in item and item[field] for field in ['question', 'input_text'])
#             has_answer = any(field in item and item[field] for field in ['answer', 'target_text'])
            
#             if not (has_question and has_answer):
#                 for field in required_fields:
#                     if not any(alt in item and item[alt] for alt in [field, f"input_{field}", f"target_{field}"]):
#                         structure_stats['missing_fields'][field] = structure_stats['missing_fields'].get(field, 0) + 1
#                 continue
            
#             structure_stats['valid_items'] += 1
            
#             # Détecter le type de structure
#             if item.get('data_source') == 'nouvelle_structure':
#                 structure_stats['has_new_structure'] = True
#             elif item.get('data_source') == 'legacy_structure':
#                 structure_stats['has_legacy_structure'] = True
            
#             # Statistiques des sources
#             source = item.get('data_source', 'unknown')
#             structure_stats['data_sources'][source] = structure_stats['data_sources'].get(source, 0) + 1
        
#         # Rapport de validation
#         print(f"   ✓ Items valides: {structure_stats['valid_items']}/{structure_stats['total_items']}")
        
#         if structure_stats['has_new_structure']:
#             print("   ✓ Structure moderne détectée (qa_pairs)")
#         if structure_stats['has_legacy_structure']:
#             print("   ✓ Structure legacy détectée")
        
#         if structure_stats['data_sources']:
#             print("   📊 Sources de données:")
#             for source, count in structure_stats['data_sources'].items():
#                 print(f"      - {source}: {count}")
        
#         if issues:
#             print(f"   ⚠️ {len(issues)} problèmes détectés (premiers items)")
#             for issue in issues[:5]:
#                 print(f"      - {issue}")
        
#         if structure_stats['valid_items'] == 0:
#             raise ValueError("Aucun item valide trouvé dans les données d'entraînement!")
        
#         return structure_stats

#     def save_enhanced_metadata(self, training_data, training_time, val_data=None, original_size=None):
#         """Sauvegarde des métadonnées enrichies pour la nouvelle structure"""
        
#         # Analyse des territoires
#         territories = {}
#         question_types = {}
#         indicators_found = set()
#         data_sources = {}
#         gender_distribution = {}
        
#         for item in training_data:
#             # Territoires
#             territory = item.get('territory', item.get('original_territory', 'Unknown'))
#             territories[territory] = territories.get(territory, 0) + 1
            
#             # Types de questions
#             qtype = item.get('question_type', 'unknown')
#             question_types[qtype] = question_types.get(qtype, 0) + 1
            
#             # Indicateurs
#             if 'indicators' in item and isinstance(item['indicators'], dict):
#                 indicators_found.update(item['indicators'].keys())
            
#             variable = item.get('variable', item.get('indicateur'))
#             if variable:
#                 indicators_found.add(variable)
            
#             # Sources de données
#             source = item.get('data_source', 'unknown')
#             data_sources[source] = data_sources.get(source, 0) + 1
            
#             # Distribution de genre
#             gender = item.get('sexe', item.get('genre', 'non spécifié'))
#             gender_distribution[gender] = gender_distribution.get(gender, 0) + 1
        
#         # Métadonnées complètes
#         metadata = {
#             'model_info': {
#                 'base_model': getattr(self.config, 'BASE_MODEL', 'unknown'),
#                 'model_type': 'enhanced_dialogue_chatbot',
#                 'training_format': 'conversational_with_context',
#                 'structure_support': {
#                     'nouvelle_structure': any(item.get('data_source') == 'nouvelle_structure' for item in training_data),
#                     'legacy_structure': any(item.get('data_source') == 'legacy_structure' for item in training_data)
#                 }
#             },
            
#             'training_config': {
#                 'num_samples_used': len(training_data),
#                 'num_samples_original': original_size or len(training_data),
#                 'validation_samples': len(val_data) if val_data else 0,
#                 'max_length': self.config.MAX_LENGTH,
#                 'num_epochs': self.config.NUM_EPOCHS,
#                 'learning_rate': self.config.LEARNING_RATE,
#                 'batch_size': self.config.BATCH_SIZE,
#                 'gradient_accumulation_steps': getattr(self.config, 'GRADIENT_ACCUMULATION_STEPS', 1),
#                 'subset_used': len(training_data) != (original_size or len(training_data))
#             },
            
#             'data_analysis': {
#                 'territories': {
#                     'count': len(territories),
#                     'distribution': dict(sorted(territories.items(), key=lambda x: x[1], reverse=True)[:15])
#                 },
#                 'question_types': {
#                     'count': len(question_types),
#                     'distribution': question_types
#                 },
#                 'indicators_available': sorted(list(indicators_found)),
#                 'data_sources': data_sources,
#                 'gender_distribution': gender_distribution
#             },
            
#             'performance': {
#                 'training_time_minutes': round(training_time, 2),
#                 'samples_per_minute': round(len(training_data) * self.config.NUM_EPOCHS / training_time, 2),
#                 'training_date': time.strftime("%Y-%m-%d %H:%M:%S"),
#                 'efficiency_score': min(100, round((len(training_data) * self.config.NUM_EPOCHS) / (training_time * 10), 1))
#             },
            
#             'hardware_info': {
#                 'cpu_count': psutil.cpu_count(),
#                 'ram_gb': round(psutil.virtual_memory().total / (1024**3), 1),
#                 'torch_threads': torch.get_num_threads(),
#                 'platform': 'CPU' if not torch.cuda.is_available() else 'GPU',
#                 'final_ram_usage': round(psutil.virtual_memory().percent, 1)
#             },
            
#             'usage_recommendations': {
#                 'best_question_types': list(sorted(question_types.keys(), key=lambda x: question_types[x], reverse=True)[:5]),
#                 'supported_territories': list(territories.keys())[:15],
#                 'optimal_query_format': "Mentionnez le territoire et soyez spécifique",
#                 'context_tokens_supported': [
#                     "<|territory|>", "<|indicator|>", "<|genre|>", 
#                     "<|user|>", "<|assistant|>"
#                 ]
#             },
            
#             'quality_metrics': {
#                 'data_coverage': {
#                     'territory_coverage': len(territories),
#                     'question_type_coverage': len(question_types),
#                     'indicator_coverage': len(indicators_found)
#                 },
#                 'data_balance': {
#                     'most_common_territory_ratio': max(territories.values()) / len(training_data) if territories else 0,
#                     'most_common_qtype_ratio': max(question_types.values()) / len(training_data) if question_types else 0
#                 }
#             }
#         }
        
#         # Sauvegarde
#         metadata_path = os.path.join(self.config.MODEL_PATH, 'enhanced_metadata.json')
#         with open(metadata_path, 'w', encoding='utf-8') as f:
#             json.dump(metadata, f, indent=2, ensure_ascii=False)
        
#         print(f"📋 Métadonnées enrichies sauvées: {metadata_path}")
#         return metadata

#     def print_enhanced_training_summary(self, training_data, training_time, original_size):
#         """Affiche un résumé détaillé de l'entraînement"""
#         print("\n" + "="*60)
#         print("🎯 RÉSUMÉ DÉTAILLÉ DE L'ENTRAÎNEMENT")
#         print("="*60)
        
#         # Analyse des données
#         territories = set(item.get('territory', 'Unknown') for item in training_data)
#         question_types = {}
#         data_sources = {}
#         gender_dist = {}
        
#         for item in training_data:
#             qtype = item.get('question_type', 'unknown')
#             question_types[qtype] = question_types.get(qtype, 0) + 1
            
#             source = item.get('data_source', 'unknown')
#             data_sources[source] = data_sources.get(source, 0) + 1
            
#             gender = item.get('sexe', 'non spécifié')
#             gender_dist[gender] = gender_dist.get(gender, 0) + 1
        
#         # Informations générales
#         print(f"📊 DONNÉES D'ENTRAÎNEMENT:")
#         print(f"   • Échantillons utilisés: {len(training_data)}")
#         if original_size and original_size != len(training_data):
#             print(f"   • Échantillons originaux: {original_size} (sous-ensemble utilisé)")
#         print(f"   • Territoires couverts: {len(territories)}")
#         print(f"   • Types de questions: {len(question_types)}")
#         print(f"   • Époques d'entraînement: {self.config.NUM_EPOCHS}")
        
#         # Sources de données
#         if len(data_sources) > 1:
#             print(f"\n🔄 SOURCES DE DONNÉES:")
#             for source, count in data_sources.items():
#                 percentage = (count / len(training_data)) * 100
#                 print(f"   • {source}: {count} ({percentage:.1f}%)")
        
#         # Performance
#         samples_total = len(training_data) * self.config.NUM_EPOCHS
#         speed = samples_total / (training_time * 60)
        
#         print(f"\n⚡ PERFORMANCE:")
#         print(f"   • Temps total: {training_time:.1f} minutes")
#         print(f"   • Vitesse: {speed:.1f} échantillons/seconde")
#         print(f"   • Échantillons traités: {samples_total:,}")
        
#         # Types de questions les plus fréquents
#         print(f"\n📈 TYPES DE QUESTIONS PRINCIPAUX:")
#         sorted_types = sorted(question_types.items(), key=lambda x: x[1], reverse=True)
#         for qtype, count in sorted_types[:6]:
#             percentage = (count / len(training_data)) * 100
#             print(f"   • {qtype}: {count} ({percentage:.1f}%)")
        
#         # Territoires principaux
#         territory_counts = {}
#         for item in training_data:
#             terr = item.get('territory', 'Unknown')
#             territory_counts[terr] = territory_counts.get(terr, 0) + 1
        
#         print(f"\n🗺️ TERRITOIRES PRINCIPAUX:")
#         sorted_territories = sorted(territory_counts.items(), key=lambda x: x[1], reverse=True)
#         for territory, count in sorted_territories[:6]:
#             percentage = (count / len(training_data)) * 100
#             display_name = territory if len(territory) <= 35 else territory[:32] + "..."
#             print(f"   • {display_name}: {count} ({percentage:.1f}%)")
        
#         # Distribution de genre si disponible
#         if any(g != 'non spécifié' for g in gender_dist.keys()):
#             print(f"\n⚧ DISTRIBUTION PAR GENRE:")
#             for gender, count in sorted(gender_dist.items(), key=lambda x: x[1], reverse=True):
#                 if gender != 'non spécifié':
#                     percentage = (count / len(training_data)) * 100
#                     print(f"   • {gender}: {count} ({percentage:.1f}%)")
        
#         # Recommandations d'utilisation
#         print(f"\n💡 RECOMMANDATIONS D'UTILISATION:")
#         print("   • Mentionnez explicitement les territoires dans vos questions")
#         print("   • Utilisez des questions spécifiques sur les indicateurs démographiques")
#         print("   • Le modèle excelle sur les données de population légale/municipale")
        
#         top_qtype = sorted_types[0][0] if sorted_types else "demographic"
#         top_territory = sorted_territories[0][0] if sorted_territories else "Maroc"
#         print(f"   • Exemple optimal: 'Quelle est la population légale de {top_territory} ?'")
        
#         # Tokens spéciaux supportés
#         print(f"\n🏷️ TOKENS DE CONTEXTE SUPPORTÉS:")
#         print("   • <|territory|> : Spécifier un territoire")
#         print("   • <|indicator|> : Spécifier un indicateur")
#         print("   • <|genre|> : Spécifier le genre")
#         print("   • <|user|> <|assistant|> : Format conversationnel")

#     def quick_test_with_new_structure(self, training_data, max_samples=500):
#         """Test rapide spécialement adapté à la nouvelle structure"""
#         print("\n🧪 TEST RAPIDE AVEC NOUVELLE STRUCTURE")
#         print("="*50)
        
#         # Validation du système
#         system_check = check_system_resources(min_ram_gb=2.0)
#         if not system_check['ready']:
#             print("❌ Système non prêt pour l'entraînement")
#             return None
        
#         # Configuration de test optimisée
#         original_config = {
#             'NUM_EPOCHS': self.config.NUM_EPOCHS,
#             'BATCH_SIZE': self.config.BATCH_SIZE,
#             'MAX_LENGTH': self.config.MAX_LENGTH
#         }
        
#         # Ajustements pour test rapide
#         self.config.NUM_EPOCHS = 1
#         self.config.BATCH_SIZE = max(1, min(self.config.BATCH_SIZE, 4))
#         self.config.MAX_LENGTH = min(self.config.MAX_LENGTH, 128)
        
#         try:
#             # Créer un sous-ensemble équilibré
#             subset = self.create_training_subset(training_data, max_samples, balanced=True)
            
#             print(f"🔧 Configuration de test:")
#             print(f"   • Échantillons: {len(subset)}")
#             print(f"   • Époques: {self.config.NUM_EPOCHS}")
#             print(f"   • Batch size: {self.config.BATCH_SIZE}")
#             print(f"   • Max length: {self.config.MAX_LENGTH}")
            
#             # Entraînement de test
#             self.train_model(subset, use_validation=False, use_subset=False)
            
#             print(f"\n✅ TEST RAPIDE TERMINÉ!")
#             print("Si les résultats sont satisfaisants, vous pouvez maintenant:")
#             print("   • Augmenter NUM_EPOCHS (2-3)")
#             print("   • Augmenter le nombre d'échantillons")
#             print("   • Entraîner sur le dataset complet")
            
#             return True
            
#         finally:
#             # Restaurer la configuration originale
#             for key, value in original_config.items():
#                 setattr(self.config, key, value)


# def check_system_resources(min_ram_gb: float = 4.0):
#     """Vérification des ressources système optimisée"""
#     memory = psutil.virtual_memory()
#     cpu_percent = psutil.cpu_percent(interval=1)
    
#     print("🔍 Vérification des ressources pour nouvelle structure:")
#     print(f"   • CPU: {psutil.cpu_count()} cœurs, utilisation: {cpu_percent:.1f}%")
#     print(f"   • RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB disponible")
#     print(f"   • RAM utilisée: {memory.percent:.1f}%")
    
#     warnings = []
#     recommendations = []
    
#     if memory.available < min_ram_gb * (1024**3):
#         warnings.append(f"Moins de {min_ram_gb}GB RAM disponible")
#         recommendations.append("Fermer applications gourmandes en mémoire")
    
#     if cpu_percent > 80:
#         warnings.append("CPU très chargé")
#         recommendations.append("Attendre que la charge CPU diminue")
    
#     if memory.percent > 85:
#         warnings.append("RAM très utilisée")
#         recommendations.append("Redémarrer la session ou réduire BATCH_SIZE")
    
#     # Recommandations spécifiques à la nouvelle structure
#     if memory.available < 8 * (1024**3):
#         recommendations.append("Nouvelle structure: Réduire MAX_LENGTH pour optimiser")
    
#     if len(recommendations) == 0:
#         recommendations.append("Système prêt pour entraîner avec nouvelle structure!")
    
#     if warnings:
#         print("⚠️ Avertissements:")
#         for warning in warnings:
#             print(f"      - {warning}")
    
#     print("💡 Recommandations:")
#     for rec in recommendations:
#         print(f"      - {rec}")
    
#     # Score de préparation ajusté
#     readiness_score = 100
#     if memory.available < min_ram_gb * (1024**3):
#         readiness_score -= 40
#     if cpu_percent > 80:
#         readiness_score -= 30
#     if memory.percent > 85:
#         readiness_score -= 20
    
#     print(f"\n📊 Score de préparation: {readiness_score}/100")
    
#     return {
#         'ready': readiness_score >= 70,
#         'score': readiness_score,
#         'warnings': warnings,
#         'recommendations': recommendations,
#         'memory_available_gb': memory.available / (1024**3),
#         'cpu_usage_percent': cpu_percent,
#         'optimized_for_new_structure': True
#     }


# def train_with_new_structure_data(config, training_data, quick_test=False):
#     """Fonction principale pour entraîner avec la nouvelle structure de données"""
#     print("🚀 ENTRAÎNEMENT AVEC NOUVELLE STRUCTURE DE DONNÉES")
#     print("="*55)
    
#     # Vérifications initiales
#     if not training_data:
#         print("❌ Aucune donnée d'entraînement fournie")
#         return None
    
#     # Vérifier la présence de la nouvelle structure
#     new_structure_count = sum(1 for item in training_data if item.get('data_source') == 'nouvelle_structure')
#     legacy_structure_count = len(training_data) - new_structure_count
    
#     print(f"📊 Analyse de la structure:")
#     print(f"   • Nouvelle structure: {new_structure_count} éléments")
#     print(f"   • Structure legacy: {legacy_structure_count} éléments")
    
#     # Initialiser le trainer
#     trainer = HCPChatbotTrainer(config)
    
#     if quick_test:
#         print(f"\n⚡ Mode test rapide activé")
#         return trainer.quick_test_with_new_structure(training_data, max_samples=1000)
#     else:
#         # Entraînement complet avec recommandations adaptées
#         print(f"\n🎯 Mode entraînement complet")
        
#         # Vérifications système
#         system_check = check_system_resources(min_ram_gb=4.0)
#         if not system_check['ready']:
#             print("⚠️ Système non optimal, mais on peut continuer avec précautions")
        
#         # Estimation et recommandations
#         estimated_time, recommend_subset = trainer.estimate_training_time(len(training_data))
        
#         # Options d'entraînement
#         use_subset = recommend_subset and len(training_data) > 10000
#         use_validation = len(training_data) > 1000
        
#         print(f"\n⚙️ Configuration d'entraînement recommandée:")
#         print(f"   • Utiliser sous-ensemble: {'Oui' if use_subset else 'Non'}")
#         print(f"   • Utiliser validation: {'Oui' if use_validation else 'Non'}")
        
#         # Lancer l'entraînement
#         trainer.train_model(
#             training_data, 
#             use_validation=use_validation,
#             use_subset=use_subset,
#             max_subset_size=8000
#         )
        
#         return trainer


# if __name__ == "__main__":
#     try:
#         from config import Config
#         from data_processor import HCPDataProcessor
        
#         print("Test du trainer avec nouvelle structure...")
        
#         # Charger les données
#         processor = HCPDataProcessor(Config)
#         data = processor.load_all_data()
        
#         if not data.empty:
#             qa_pairs = processor.create_qa_pairs()
            
#             # Test rapide
#             train_with_new_structure_data(Config, qa_pairs, quick_test=True)
#         else:
#             print("❌ Aucune donnée chargée pour le test")
            
#     except ImportError:
#         print("❌ Modules requis non disponibles. Assurez-vous que config.py et data_processor.py existent.")










# import torch
# from transformers import (
#     AutoTokenizer, AutoModelForCausalLM, 
#     TrainingArguments, Trainer, 
#     DataCollatorForLanguageModeling
# )
# from datasets import Dataset
# import json
# import os
# import time
# import psutil
# import gc

# class HCPChatbotTrainer:
#     def __init__(self, config):
#         self.config = config
#         self.tokenizer = None
#         self.model = None
#         self.hf_token = getattr(config, 'HF_TOKEN', None) or os.getenv('HUGGING_FACE_TOKEN')
#         self.setup_torch_optimizations()

#     def setup_torch_optimizations(self):
#         """Configuration optimale pour PyTorch"""
#         torch.set_num_threads(8)
#         if not torch.cuda.is_available():
#             try:
#                 torch.backends.mkl.enabled = True
#             except:
#                 pass
        
#         print(f"Threads PyTorch: {torch.get_num_threads()}")
#         print(f"RAM disponible: {psutil.virtual_memory().available / (1024**3):.1f} GB")

#     def initialize_model(self):
#         """Initialisation du modèle et tokenizer avec support nouvelle structure HCP"""
#         print(f"Initialisation du modèle {self.config.BASE_MODEL}...")
        
#         # Tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.config.BASE_MODEL,
#             padding_side="left",
#             token=self.hf_token
#         )
        
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
        
#         # Tokens spéciaux adaptés à la nouvelle structure HCP
#         special_tokens = []
#         required_tokens = [
#             "<|user|>", "<|assistant|>", 
#             "<|territory|>", "<|indicator|>", "<|genre|>",
#             "<|source|>", "<|context|>", "<|hcp|>"
#         ]
        
#         for token in required_tokens:
#             if token not in self.tokenizer.get_vocab():
#                 special_tokens.append(token)
        
#         if special_tokens:
#             self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
#             print(f"Tokens spéciaux HCP ajoutés: {special_tokens}")

#         # Modèle
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.config.BASE_MODEL,
#             torch_dtype=torch.float32,
#             low_cpu_mem_usage=True,
#             token=self.hf_token
#         )
        
#         if special_tokens:
#             self.model.resize_token_embeddings(len(self.tokenizer))
#             print("Embeddings redimensionnés pour les tokens HCP")
        
#         print(f"Modèle {self.config.BASE_MODEL} initialisé avec succès")
#         print(f"Paramètres du modèle: {self.model.num_parameters():,}")

#     def format_dialogue_for_training(self, data_item):
#         """Formate les dialogues pour l'entraînement avec contexte HCP enrichi"""
        
#         # Si c'est déjà une conversation formatée
#         if 'conversation' in data_item and data_item['conversation']:
#             return data_item['conversation']
        
#         # Extraction des données
#         question = data_item.get('input_text', data_item.get('question', ''))
#         answer = data_item.get('target_text', data_item.get('answer', ''))
        
#         # Métadonnées de la nouvelle structure HCP
#         territory = data_item.get('territory', data_item.get('original_territory', ''))
#         indicator = data_item.get('variable', data_item.get('indicateur', ''))
#         genre = data_item.get('sexe', data_item.get('genre', ''))
#         source = data_item.get('source_data', data_item.get('source', ''))
        
#         # Format conditionnel selon les métadonnées disponibles
#         context_parts = []
        
#         # Toujours commencer par le contexte HCP
#         context_parts.append("<|hcp|>")
        
#         # Territoire (très important dans les données HCP)
#         if territory and territory not in ['Territoire non spécifié', '']:
#             context_parts.append(f"<|territory|>{territory}")
        
#         # Indicateur spécifique HCP
#         if indicator and indicator not in ['indicateur_demographique', 'indicateur démographique', '']:
#             context_parts.append(f"<|indicator|>{indicator}")
            
#         # Genre (important pour les stats démographiques)
#         if genre and genre not in ['non spécifié', '']:
#             context_parts.append(f"<|genre|>{genre}")
            
#         # Source des données
#         if source and source not in ['unknown', '']:
#             context_parts.append(f"<|source|>{source}")
        
#         # Construction du dialogue final optimisé pour HCP
#         context_prefix = "".join(context_parts)
#         formatted = f"{context_prefix}<|user|>{question}<|assistant|>{answer}<|endoftext|>"
        
#         return formatted

#     def tokenize_data(self, examples):
#         """Tokenisation des données avec support du nouveau format HCP"""
#         conversations = []
        
#         # Gérer la nouvelle structure HCP
#         for i in range(len(examples['input_text'])):
#             data_item = {
#                 'input_text': examples['input_text'][i],
#                 'target_text': examples['target_text'][i],
#                 'territory': examples.get('territory', [''])[i] if 'territory' in examples else '',
#                 'variable': examples.get('variable', [''])[i] if 'variable' in examples else '',
#                 'sexe': examples.get('sexe', [''])[i] if 'sexe' in examples else '',
#                 'source_data': examples.get('source_data', [''])[i] if 'source_data' in examples else '',
#                 'question_type': examples.get('question_type', [''])[i] if 'question_type' in examples else ''
#             }
            
#             formatted_conversation = self.format_dialogue_for_training(data_item)
#             conversations.append(formatted_conversation)
        
#         # Tokenisation avec longueur adaptée aux données HCP
#         inputs = self.tokenizer(
#             conversations,
#             truncation=True,
#             padding='max_length',
#             max_length=self.config.MAX_LENGTH,
#             return_tensors=None
#         )
        
#         inputs["labels"] = inputs["input_ids"].copy()
#         return inputs

#     def prepare_dataset(self, training_data):
#         """Préparation du dataset avec validation renforcée pour HCP"""
#         print(f"Préparation du dataset HCP avec {len(training_data)} échantillons...")
        
#         # Nettoyage spécifique à la structure HCP
#         cleaned_data = []
#         for item in training_data:
#             cleaned_item = {}
            
#             # Mapping des champs spécifiques HCP
#             field_mapping = {
#                 'input_text': ['input_text', 'question'],
#                 'target_text': ['target_text', 'answer'],
#                 'territory': ['territory', 'original_territory', 'territoire'],
#                 'variable': ['variable', 'indicator', 'indicateur'],
#                 'sexe': ['sexe', 'genre'],
#                 'source_data': ['source_data', 'source'],
#                 'question_type': ['question_type'],
#                 'data_source': ['data_source']
#             }
            
#             for target_field, source_fields in field_mapping.items():
#                 value = None
#                 for source_field in source_fields:
#                     if source_field in item and item[source_field]:
#                         value = item[source_field]
#                         break
                
#                 # Nettoyage et normalisation des valeurs HCP
#                 if isinstance(value, (list, dict, tuple)):
#                     cleaned_item[target_field] = str(value) if value else ""
#                 elif value is None:
#                     cleaned_item[target_field] = ""
#                 else:
#                     cleaned_item[target_field] = str(value).strip()
                    
#                 # Normalisation spécifique HCP
#                 if target_field == 'sexe' and cleaned_item[target_field] in ['', 'non spécifié']:
#                     cleaned_item[target_field] = 'ensemble'
#                 elif target_field == 'territory' and cleaned_item[target_field] == '':
#                     cleaned_item[target_field] = 'Territoire non spécifié'
            
#             # Validation des champs obligatoires
#             if cleaned_item.get('input_text') and cleaned_item.get('target_text'):
#                 cleaned_data.append(cleaned_item)
        
#         print(f"  ✓ {len(cleaned_data)} échantillons HCP valides après nettoyage")
        
#         if not cleaned_data:
#             raise ValueError("Aucune donnée HCP valide après nettoyage!")
        
#         # Affichage des statistiques de nettoyage
#         territory_stats = {}
#         indicator_stats = {}
#         for item in cleaned_data:
#             territory = item.get('territory', 'Unknown')
#             territory_stats[territory] = territory_stats.get(territory, 0) + 1
            
#             indicator = item.get('variable', 'Unknown')
#             indicator_stats[indicator] = indicator_stats.get(indicator, 0) + 1
        
#         print(f"  📊 Territoires: {len(territory_stats)}, Indicateurs: {len(indicator_stats)}")
        
#         # Création du dataset
#         dataset = Dataset.from_list(cleaned_data)
        
#         # Tokenisation
#         tokenized_dataset = dataset.map(
#             self.tokenize_data,
#             batched=True,
#             num_proc=min(4, len(cleaned_data) // 100 + 1),
#             remove_columns=dataset.column_names,
#             desc="Tokenisation des données HCP avec contexte enrichi"
#         )
        
#         print(f"Dataset HCP tokenisé: {len(tokenized_dataset)} échantillons")
#         return tokenized_dataset

#     def estimate_training_time(self, num_samples):
#         """Estimation du temps d'entraînement pour données HCP"""
#         # Facteurs d'ajustement spécifiques HCP
#         base_samples_per_second = 1.2  # Les données HCP sont plus complexes
        
#         if self.config.MAX_LENGTH > 512:
#             base_samples_per_second *= 0.7
#         if self.config.BATCH_SIZE > 2:
#             base_samples_per_second *= 1.1
        
#         # Les données HCP avec contexte enrichi sont plus lourdes
#         base_samples_per_second *= 0.8
        
#         total_samples = num_samples * self.config.NUM_EPOCHS
#         estimated_time = total_samples / base_samples_per_second / 60
        
#         print(f"Temps d'entraînement HCP estimé: {estimated_time:.1f} minutes ({estimated_time/60:.1f} heures)")
        
#         return estimated_time, False  # CORRECTION: Ne jamais recommander automatiquement un sous-ensemble

#     def create_validation_split(self, training_data, val_ratio=0.1):
#         """Création du split validation équilibré pour HCP - VERSION CORRIGÉE"""
#         if val_ratio <= 0:
#             return training_data, []
        
#         import random
        
#         # Split équilibré par territoire ET indicateur (critères HCP importants)
#         territory_indicator_groups = {}
        
#         for i, item in enumerate(training_data):
#             territory = item.get('territory', 'Unknown')
#             indicator = item.get('variable', 'unknown')
#             key = f"{territory}_{indicator}"
            
#             if key not in territory_indicator_groups:
#                 territory_indicator_groups[key] = []
#             territory_indicator_groups[key].append(i)
        
#         train_indices = []
#         val_indices = []
        
#         # CORRECTION DU BUG : Mélanger et diviser correctement chaque groupe
#         for indices in territory_indicator_groups.values():
#             # Mélanger les indices pour avoir un split aléatoire
#             shuffled_indices = indices.copy()
#             random.shuffle(shuffled_indices)
            
#             # Calculer le nombre d'échantillons pour validation
#             val_size = max(1, int(len(shuffled_indices) * val_ratio))
            
#             # CORRECTION : val d'abord, puis train avec le reste
#             val_indices.extend(shuffled_indices[:val_size])
#             train_indices.extend(shuffled_indices[val_size:])  # Le reste pour train
        
#         train_data = [training_data[i] for i in train_indices]
#         val_data = [training_data[i] for i in val_indices]
        
#         print(f"Split validation HCP équilibré: {len(train_data)} train, {len(val_data)} validation")
        
#         # Validation du split
#         if len(train_data) == 0:
#             print("⚠️ ERREUR: Aucune donnée d'entraînement après split!")
#             print("Fallback: utilisation d'un split simple")
            
#             # Split simple en fallback
#             total_size = len(training_data)
#             val_size = int(total_size * val_ratio)
            
#             shuffled_data = training_data.copy()
#             random.shuffle(shuffled_data)
            
#             val_data = shuffled_data[:val_size]
#             train_data = shuffled_data[val_size:]
            
#             print(f"Split simple appliqué: {len(train_data)} train, {len(val_data)} validation")
        
#         # Statistiques de validation HCP
#         val_territories = set(item.get('territory') for item in val_data)
#         val_indicators = set(item.get('variable') for item in val_data)
#         train_territories = set(item.get('territory') for item in train_data)
#         train_indicators = set(item.get('variable') for item in train_data)
        
#         print(f"  Train couvre: {len(train_territories)} territoires, {len(train_indicators)} indicateurs")
#         print(f"  Validation couvre: {len(val_territories)} territoires, {len(val_indicators)} indicateurs")
        
#         return train_data, val_data

#     def calculate_compatible_steps(self, dataset_size, batch_size, num_epochs, use_validation=False):
#         """Calcul des steps compatibles pour éviter les erreurs - optimisé HCP"""
#         gradient_accumulation_steps = getattr(self.config, 'GRADIENT_ACCUMULATION_STEPS', 1)
        
#         # Steps par époque
#         steps_per_epoch = dataset_size // (batch_size * gradient_accumulation_steps)
#         if dataset_size % (batch_size * gradient_accumulation_steps) != 0:
#             steps_per_epoch += 1
        
#         # Total steps
#         total_steps = steps_per_epoch * num_epochs
        
#         if use_validation:
#             # Eval steps plus fréquents pour données HCP (monitoring important)
#             target_eval_ratio = 0.15
#             target_eval_steps = max(30, int(steps_per_epoch * target_eval_ratio))
            
#             eval_steps = target_eval_steps
#             while steps_per_epoch % eval_steps != 0 and eval_steps > 5:
#                 eval_steps -= 1
            
#             if eval_steps <= 5:
#                 eval_steps = steps_per_epoch
            
#             save_steps = eval_steps
#         else:
#             eval_steps = None
#             save_steps = min(steps_per_epoch, 300)
        
#         # Logging plus fréquent pour surveiller l'entraînement HCP
#         logging_steps = max(5, min(steps_per_epoch // 15, 50))
        
#         print(f"Configuration des steps pour données HCP:")
#         print(f"   - Dataset HCP: {dataset_size} échantillons")
#         print(f"   - Batch effectif: {batch_size * gradient_accumulation_steps}")
#         print(f"   - Steps/époque: {steps_per_epoch}")
#         print(f"   - Total steps: {total_steps}")
#         print(f"   - Logging steps: {logging_steps} (monitoring HCP renforcé)")
        
#         return {
#             'total_steps': total_steps,
#             'steps_per_epoch': steps_per_epoch,
#             'logging_steps': logging_steps,
#             'save_steps': save_steps,
#             'eval_steps': eval_steps
#         }

#     def train_model(self, training_data, use_validation=False, force_use_all_data=False):
#         """Entraînement principal avec support de la structure HCP - VERSION MODIFIÉE"""
#         start_time = time.time()
        
#         if self.model is None:
#             self.initialize_model()
        
#         original_size = len(training_data)
        
#         # Validation de la structure des données HCP
#         self._validate_hcp_training_data(training_data)
        
#         # CORRECTION MAJEURE: Estimation sans forcer le sous-ensemble
#         estimated_time, _ = self.estimate_training_time(len(training_data))
        
#         # CORRECTION: Afficher l'estimation mais ne pas forcer le sous-ensemble
#         if estimated_time > 90:
#             print(f"⏰ Estimation: {estimated_time:.1f} minutes d'entraînement")
#             print("💡 Conseils d'optimisation disponibles mais entraînement sur toutes les données...")
        
#         # FORCE L'UTILISATION DE TOUTES LES DONNÉES si demandé
#         if force_use_all_data:
#             print(f"🚀 ENTRAÎNEMENT FORCÉ SUR TOUTES LES DONNÉES: {len(training_data)} échantillons")
        
#         # Split validation
#         if use_validation:
#             train_data, val_data = self.create_validation_split(training_data, 0.1)
#         else:
#             train_data = training_data
#             val_data = []
        
#         # Préparation des datasets
#         train_dataset = self.prepare_dataset(train_data)
#         val_dataset = self.prepare_dataset(val_data) if val_data else None
        
#         # Création du répertoire modèle
#         os.makedirs(self.config.MODEL_PATH, exist_ok=True)
        
#         # Configuration des steps
#         steps_config = self.calculate_compatible_steps(
#             len(train_dataset), 
#             self.config.BATCH_SIZE, 
#             self.config.NUM_EPOCHS,
#             use_validation and val_dataset is not None
#         )
        
#         # Arguments d'entraînement optimisés pour HCP
#         training_args = TrainingArguments(
#             output_dir=self.config.MODEL_PATH,
#             overwrite_output_dir=True,
            
#             # Époques et steps
#             num_train_epochs=self.config.NUM_EPOCHS,
#             max_steps=-1,
            
#             # Batch et apprentissage (ajusté pour données HCP)
#             per_device_train_batch_size=self.config.BATCH_SIZE,
#             gradient_accumulation_steps=getattr(self.config, 'GRADIENT_ACCUMULATION_STEPS', 1),
#             learning_rate=self.config.LEARNING_RATE,
#             weight_decay=getattr(self.config, 'WEIGHT_DECAY', 0.01),
#             warmup_steps=max(10, steps_config['total_steps'] // 20),  # Warmup pour HCP
            
#             # Dataloaders
#             dataloader_num_workers=min(4, getattr(self.config, 'DATALOADER_NUM_WORKERS', 2)),
#             dataloader_pin_memory=False,
#             remove_unused_columns=False,
            
#             # Logging et sauvegarde (plus fréquent pour HCP)
#             logging_steps=steps_config['logging_steps'],
#             save_steps=steps_config['save_steps'],
#             save_total_limit=3,  # Plus de sauvegardes pour HCP
            
#             # Évaluation
#             evaluation_strategy="steps" if steps_config['eval_steps'] else "no",
#             eval_steps=steps_config['eval_steps'],
            
#             # Optimisations
#             prediction_loss_only=True,
#             report_to=None,
#             load_best_model_at_end=bool(steps_config['eval_steps']),
#             metric_for_best_model="eval_loss" if steps_config['eval_steps'] else None,
#             gradient_checkpointing=True,
#             bf16=False,
#             fp16=False,
#         )
        
#         # Data collator
#         data_collator = DataCollatorForLanguageModeling(
#             tokenizer=self.tokenizer,
#             mlm=False,
#         )
        
#         # Trainer
#         trainer = Trainer(
#             model=self.model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=val_dataset,
#             data_collator=data_collator,
#         )
        
#         # Entraînement
#         print(f"\n🚀 DÉBUT DE L'ENTRAÎNEMENT HCP SUR TOUTES LES DONNÉES")
#         print(f"Structure: Données HCP avec contexte enrichi")
#         print(f"Configuration: {self.config.NUM_EPOCHS} époques avec monitoring renforcé")
#         print(f"Échantillons d'entraînement: {len(train_dataset):,}")
#         if val_dataset:
#             print(f"Échantillons de validation: {len(val_dataset):,}")
        
#         try:
#             memory_before = psutil.virtual_memory()
#             print(f"RAM avant: {memory_before.percent:.1f}%")
            
#             gc.collect()
#             trainer.train()
            
#             # Sauvegarde
#             print("💾 Sauvegarde du modèle HCP...")
#             trainer.save_model()
#             self.tokenizer.save_pretrained(self.config.MODEL_PATH)
            
#             training_time = (time.time() - start_time) / 60
            
#             # Métadonnées enrichies HCP
#             self.save_hcp_enhanced_metadata(train_data, training_time, val_data, original_size)
            
#             print(f"\n✅ ENTRAÎNEMENT HCP TERMINÉ EN {training_time:.1f} minutes!")
#             print(f"📁 Modèle HCP sauvé: {self.config.MODEL_PATH}")
            
#             self.print_hcp_training_summary(train_data, training_time, original_size)
            
#         except Exception as e:
#             print(f"❌ Erreur pendant l'entraînement HCP: {e}")
#             raise
#         finally:
#             gc.collect()

#     def _validate_hcp_training_data(self, training_data):
#         """Valide la structure des données d'entraînement HCP"""
#         if not training_data:
#             raise ValueError("Aucune donnée d'entraînement HCP fournie!")
        
#         print("🔍 Validation de la structure des données HCP...")
        
#         required_fields = ['question', 'answer']
#         hcp_fields = ['territory', 'variable', 'source_data', 'sexe']
        
#         issues = []
#         structure_stats = {
#             'total_items': len(training_data),
#             'valid_items': 0,
#             'missing_fields': {},
#             'hcp_coverage': {},
#             'has_new_structure': False,
#             'territory_count': 0,
#             'indicator_count': 0,
#             'sources': set()
#         }
        
#         territories = set()
#         indicators = set()
        
#         for i, item in enumerate(training_data[:100]):  # Vérifier les 100 premiers
#             if not isinstance(item, dict):
#                 issues.append(f"Item {i}: n'est pas un dictionnaire")
#                 continue
            
#             # Vérifier les champs requis
#             has_question = any(field in item and item[field] for field in ['question', 'input_text'])
#             has_answer = any(field in item and item[field] for field in ['answer', 'target_text'])
            
#             if not (has_question and has_answer):
#                 for field in required_fields:
#                     if not any(alt in item and item[alt] for alt in [field, f"input_{field}", f"target_{field}"]):
#                         structure_stats['missing_fields'][field] = structure_stats['missing_fields'].get(field, 0) + 1
#                 continue
            
#             structure_stats['valid_items'] += 1
            
#             # Analyser les champs HCP spécifiques
#             territory = item.get('territory', item.get('original_territory', ''))
#             if territory and territory != 'Territoire non spécifié':
#                 territories.add(territory)
#                 structure_stats['hcp_coverage']['territory'] = structure_stats['hcp_coverage'].get('territory', 0) + 1
            
#             indicator = item.get('variable', item.get('indicateur', ''))
#             if indicator and indicator not in ['indicateur_demographique', '']:
#                 indicators.add(indicator)
#                 structure_stats['hcp_coverage']['indicator'] = structure_stats['hcp_coverage'].get('indicator', 0) + 1
            
#             source = item.get('source_data', item.get('source', ''))
#             if source:
#                 structure_stats['sources'].add(source)
            
#             # Détecter le type de structure
#             if item.get('data_source') == 'nouvelle_structure':
#                 structure_stats['has_new_structure'] = True
        
#         structure_stats['territory_count'] = len(territories)
#         structure_stats['indicator_count'] = len(indicators)
        
#         # Rapport de validation HCP
#         print(f"   ✓ Items HCP valides: {structure_stats['valid_items']}/{structure_stats['total_items']}")
#         print(f"   📍 Territoires détectés: {structure_stats['territory_count']}")
#         print(f"   📊 Indicateurs détectés: {structure_stats['indicator_count']}")
        
#         if structure_stats['sources']:
#             print(f"   🔄 Sources HCP: {', '.join(list(structure_stats['sources'])[:5])}")
        
#         if structure_stats['has_new_structure']:
#             print("   ✓ Structure moderne HCP détectée")
        
#         # Afficher les territoires principaux
#         if territories:
#             top_territories = list(territories)[:5]
#             print(f"   🏛️ Territoires principaux: {', '.join(t[:30] + '...' if len(t) > 30 else t for t in top_territories)}")
        
#         # Afficher les indicateurs principaux  
#         if indicators:
#             top_indicators = list(indicators)[:5]
#             print(f"   📈 Indicateurs principaux: {', '.join(top_indicators)}")
        
#         if issues:
#             print(f"   ⚠️ {len(issues)} problèmes détectés (premiers items)")
#             for issue in issues[:3]:
#                 print(f"      - {issue}")
        
#         if structure_stats['valid_items'] == 0:
#             raise ValueError("Aucun item HCP valide trouvé dans les données d'entraînement!")
        
#         # Validation spécifique HCP
#         if structure_stats['territory_count'] < 2:
#             print("   ⚠️ Peu de territoires détectés - vérifiez la qualité des données")
        
#         if structure_stats['indicator_count'] < 5:
#             print("   ⚠️ Peu d'indicateurs détectés - vérifiez la diversité des données")
        
#         return structure_stats

#     def save_hcp_enhanced_metadata(self, training_data, training_time, val_data=None, original_size=None):
#         """Sauvegarde des métadonnées enrichies spécifiques HCP"""
        
#         # Analyse spécifique HCP
#         territories = {}
#         indicators = {}
#         sources = {}
#         gender_distribution = {}
#         question_types = {}
        
#         for item in training_data:
#             # Territoires HCP
#             territory = item.get('territory', item.get('original_territory', 'Unknown'))
#             territories[territory] = territories.get(territory, 0) + 1
            
#             # Indicateurs HCP
#             indicator = item.get('variable', item.get('indicateur', 'unknown'))
#             indicators[indicator] = indicators.get(indicator, 0) + 1
            
#             # Sources HCP
#             source = item.get('source_data', item.get('source', 'unknown'))
#             sources[source] = sources.get(source, 0) + 1
            
#             # Distribution de genre
#             gender = item.get('sexe', item.get('genre', 'ensemble'))
#             gender_distribution[gender] = gender_distribution.get(gender, 0) + 1
            
#             # Types de questions
#             qtype = item.get('question_type', 'unknown')
#             question_types[qtype] = question_types.get(qtype, 0) + 1
        
#         # Métadonnées HCP complètes
#         metadata = {
#             'model_info': {
#                 'base_model': getattr(self.config, 'BASE_MODEL', 'unknown'),
#                 'model_type': 'hcp_demographic_chatbot',
#                 'training_format': 'conversational_with_hcp_context',
#                 'hcp_structure_support': True,
#                 'specialized_domain': 'moroccan_demographics',
#                 'trained_on_full_dataset': True  # AJOUT: Marqueur pour toutes les données
#             },
            
#             'training_config': {
#                 'num_samples_used': len(training_data),
#                 'num_samples_original': original_size or len(training_data),
#                 'validation_samples': len(val_data) if val_data else 0,
#                 'max_length': self.config.MAX_LENGTH,
#                 'num_epochs': self.config.NUM_EPOCHS,
#                 'learning_rate': self.config.LEARNING_RATE,
#                 'batch_size': self.config.BATCH_SIZE,
#                 'gradient_accumulation_steps': getattr(self.config, 'GRADIENT_ACCUMULATION_STEPS', 1),
#                 'subset_used': False,  # CORRECTION: Marquer comme faux
#                 'full_dataset_training': True,  # AJOUT: Marqueur explicite
#                 'hcp_optimized': True
#             },
            
#             'hcp_data_analysis': {
#                 'territories': {
#                     'count': len(territories),
#                     'distribution': dict(sorted(territories.items(), key=lambda x: x[1], reverse=True)[:20]),
#                     'coverage_type': 'national_and_regional'
#                 },
#                 'indicators': {
#                     'count': len(indicators),
#                     'distribution': dict(sorted(indicators.items(), key=lambda x: x[1], reverse=True)[:15]),
#                     'main_categories': ['population_legale', 'population_municipale', 'demographics', 'employment', 'education']
#                 },
#                 'data_sources': {
#                     'count': len(sources),
#                     'distribution': sources,
#                     'primary_source': 'HCP_Morocco'
#                 },
#                 'gender_coverage': gender_distribution,
#                 'question_types': question_types
#             },
            
#             'performance': {
#                 'training_time_minutes': round(training_time, 2),
#                 'samples_per_minute': round(len(training_data) * self.config.NUM_EPOCHS / training_time, 2),
#                 'training_date': time.strftime("%Y-%m-%d %H:%M:%S"),
#                 'efficiency_score': min(100, round((len(training_data) * self.config.NUM_EPOCHS) / (training_time * 10), 1)),
#                 'hcp_complexity_factor': 1.2  # Les données HCP sont plus complexes
#             },
            
#             'hardware_info': {
#                 'cpu_count': psutil.cpu_count(),
#                 'ram_gb': round(psutil.virtual_memory().total / (1024**3), 1),
#                 'torch_threads': torch.get_num_threads(),
#                 'platform': 'CPU' if not torch.cuda.is_available() else 'GPU',
#                 'final_ram_usage': round(psutil.virtual_memory().percent, 1),
#                 'optimized_for_hcp': True
#             },
            
#             'usage_recommendations': {
#                 'best_territories': list(sorted(territories.keys(), key=lambda x: territories[x], reverse=True)[:10]),
#                 'best_indicators': list(sorted(indicators.keys(), key=lambda x: indicators[x], reverse=True)[:8]),
#                 'optimal_query_format': "Mentionnez territoire + indicateur spécifique HCP",
#                 'context_tokens_supported': [
#                     "<|hcp|>", "<|territory|>", "<|indicator|>", "<|genre|>", 
#                     "<|source|>", "<|user|>", "<|assistant|>"
#                 ],
#                 'example_queries': [
#                     "Quelle est la population légale de Casablanca ?",
#                     "Pourcentage population masculine Ensemble du territoire national ?",
#                     "Population municipale Rabat ensemble ?"
#                 ]
#             },
            
#             'quality_metrics': {
#                 'data_coverage': {
#                     'territory_coverage': len(territories),
#                     'indicator_coverage': len(indicators),
#                     'source_coverage': len(sources),
#                     'gender_coverage': len(gender_distribution)
#                 },
#                 'data_balance': {
#                     'most_common_territory_ratio': max(territories.values()) / len(training_data) if territories else 0,
#                     'most_common_indicator_ratio': max(indicators.values()) / len(training_data) if indicators else 0,
#                     'gender_balance_score': min(gender_distribution.values()) / max(gender_distribution.values()) if len(gender_distribution) > 1 else 1.0
#                 },
#                 'hcp_specialization_score': min(100, len(indicators) * len(territories) / 100)
#             }
#         }
        
#         # Sauvegarde
#         metadata_path = os.path.join(self.config.MODEL_PATH, 'hcp_enhanced_metadata.json')
#         with open(metadata_path, 'w', encoding='utf-8') as f:
#             json.dump(metadata, f, indent=2, ensure_ascii=False)
        
#         print(f"📋 Métadonnées HCP enrichies sauvées: {metadata_path}")
#         return metadata

#     def print_hcp_training_summary(self, training_data, training_time, original_size):
#         """Affiche un résumé détaillé de l'entraînement HCP"""
#         print("\n" + "="*70)
#         print("🎯 RÉSUMÉ DÉTAILLÉ DE L'ENTRAÎNEMENT HCP - TOUTES DONNÉES")
#         print("="*70)
        
#         # Analyse des données HCP
#         territories = {}
#         indicators = {}
#         sources = {}
#         gender_dist = {}
        
#         for item in training_data:
#             # Territoires
#             territory = item.get('territory', item.get('original_territory', 'Unknown'))
#             territories[territory] = territories.get(territory, 0) + 1
            
#             # Indicateurs
#             indicator = item.get('variable', item.get('indicateur', 'unknown'))
#             indicators[indicator] = indicators.get(indicator, 0) + 1
            
#             # Sources
#             source = item.get('source_data', item.get('source', 'unknown'))
#             sources[source] = sources.get(source, 0) + 1
            
#             # Genre
#             gender = item.get('sexe', item.get('genre', 'ensemble'))
#             gender_dist[gender] = gender_dist.get(gender, 0) + 1
        
#         # Informations générales HCP
#         print(f"📊 DONNÉES D'ENTRAÎNEMENT HCP (TOUTES UTILISÉES):")
#         print(f"   • Échantillons utilisés: {len(training_data):,}")
#         if original_size and original_size != len(training_data):
#             print(f"   • Échantillons originaux: {original_size:,}")
#         else:
#             print(f"   • Échantillons originaux: {len(training_data):,} (100% utilisé)")
#         print(f"   • Territoires HCP couverts: {len(territories)}")
#         print(f"   • Indicateurs démographiques: {len(indicators)}")
#         print(f"   • Sources de données: {len(sources)}")
#         print(f"   • Époques d'entraînement: {self.config.NUM_EPOCHS}")
        
#         # Sources de données HCP
#         if len(sources) > 1:
#             print(f"\n🔄 SOURCES DE DONNÉES HCP:")
#             for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
#                 percentage = (count / len(training_data)) * 100
#                 print(f"   • {source}: {count:,} ({percentage:.1f}%)")
        
#         # Performance
#         samples_total = len(training_data) * self.config.NUM_EPOCHS
#         speed = samples_total / (training_time * 60)
        
#         print(f"\n⚡ PERFORMANCE D'ENTRAÎNEMENT:")
#         print(f"   • Temps total: {training_time:.1f} minutes")
#         print(f"   • Vitesse: {speed:.1f} échantillons/seconde")
#         print(f"   • Échantillons traités: {samples_total:,}")
#         print(f"   • Complexité HCP: Élevée (contexte enrichi)")
        
#         # Indicateurs HCP principaux
#         print(f"\n📈 INDICATEURS DÉMOGRAPHIQUES PRINCIPAUX:")
#         sorted_indicators = sorted(indicators.items(), key=lambda x: x[1], reverse=True)
#         for indicator, count in sorted_indicators[:8]:
#             percentage = (count / len(training_data)) * 100
#             print(f"   • {indicator}: {count:,} ({percentage:.1f}%)")
        
#         # Territoires HCP principaux
#         print(f"\n🗺️ TERRITOIRES HCP PRINCIPAUX:")
#         sorted_territories = sorted(territories.items(), key=lambda x: x[1], reverse=True)
#         for territory, count in sorted_territories[:8]:
#             percentage = (count / len(training_data)) * 100
#             display_name = territory if len(territory) <= 45 else territory[:42] + "..."
#             print(f"   • {display_name}: {count:,} ({percentage:.1f}%)")
        
#         # Distribution de genre HCP
#         if len(gender_dist) > 1:
#             print(f"\n⚧ DISTRIBUTION PAR GENRE (HCP):")
#             for gender, count in sorted(gender_dist.items(), key=lambda x: x[1], reverse=True):
#                 percentage = (count / len(training_data)) * 100
#                 print(f"   • {gender}: {count:,} ({percentage:.1f}%)")
        
#         # Recommandations d'utilisation HCP
#         print(f"\n💡 RECOMMANDATIONS D'UTILISATION HCP:")
#         print("   • Mentionnez explicitement les territoires dans vos questions")
#         print("   • Spécifiez l'indicateur démographique recherché")
#         print("   • Le modèle excelle sur les données HCP officielles")
#         print("   • Utilisez les termes exacts HCP pour de meilleurs résultats")
        
#         # Exemples optimaux
#         top_territory = sorted_territories[0][0] if sorted_territories else "Ensemble du territoire national"
#         top_indicator = sorted_indicators[0][0] if sorted_indicators else "population_legale"
        
#         print(f"\n🎯 EXEMPLES DE REQUÊTES OPTIMALES:")
#         print(f"   • 'Quelle est la {top_indicator.replace('_', ' ')} de {top_territory} ?'")
#         print(f"   • 'Population légale {top_territory} ensemble'")
#         print(f"   • 'Pourcentage population masculine {top_territory}'")
        
#         # Tokens spéciaux HCP
#         print(f"\n🏷️ TOKENS DE CONTEXTE HCP SUPPORTÉS:")
#         print("   • <|hcp|> : Contexte HCP général")
#         print("   • <|territory|> : Spécifier un territoire")
#         print("   • <|indicator|> : Spécifier un indicateur démographique")
#         print("   • <|genre|> : Spécifier le genre (masculin/féminin/ensemble)")
#         print("   • <|source|> : Source des données HCP")
#         print("   • <|user|> <|assistant|> : Format conversationnel")
        
#         print(f"\n✅ Modèle HCP entraîné sur TOUTES LES DONNÉES - Prêt pour production!")


# def check_system_resources_hcp(min_ram_gb: float = 4.0):
#     """Vérification des ressources système optimisée pour HCP"""
#     memory = psutil.virtual_memory()
#     cpu_percent = psutil.cpu_percent(interval=1)
    
#     print("🔍 Vérification des ressources pour données HCP:")
#     print(f"   • CPU: {psutil.cpu_count()} cœurs, utilisation: {cpu_percent:.1f}%")
#     print(f"   • RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB disponible")
#     print(f"   • RAM utilisée: {memory.percent:.1f}%")
    
#     warnings = []
#     recommendations = []
    
#     if memory.available < min_ram_gb * (1024**3):
#         warnings.append(f"Moins de {min_ram_gb}GB RAM disponible")
#         recommendations.append("Fermer applications gourmandes en mémoire")
    
#     if cpu_percent > 80:
#         warnings.append("CPU très chargé")
#         recommendations.append("Attendre que la charge CPU diminue")
    
#     if memory.percent > 85:
#         warnings.append("RAM très utilisée")
#         recommendations.append("Redémarrer la session ou réduire BATCH_SIZE")
    
#     # Recommandations spécifiques HCP
#     if memory.available < 8 * (1024**3):
#         recommendations.append("Données HCP: Réduire MAX_LENGTH pour optimiser")
#         recommendations.append("Données HCP: Considérer un entraînement par batch")
    
#     if len(recommendations) == 0:
#         recommendations.append("Système prêt pour entraîner avec toutes les données HCP!")
    
#     if warnings:
#         print("⚠️ Avertissements:")
#         for warning in warnings:
#             print(f"      - {warning}")
    
#     print("💡 Recommandations HCP:")
#     for rec in recommendations:
#         print(f"      - {rec}")
    
#     # Score de préparation ajusté pour HCP
#     readiness_score = 100
#     if memory.available < min_ram_gb * (1024**3):
#         readiness_score -= 25  # Moins pénalisant pour permettre l'entraînement
#     if cpu_percent > 80:
#         readiness_score -= 20
#     if memory.percent > 85:
#         readiness_score -= 15
    
#     print(f"\n📊 Score de préparation HCP: {readiness_score}/100")
    
#     return {
#         'ready': readiness_score >= 60,  # Seuil plus permissif
#         'score': readiness_score,
#         'warnings': warnings,
#         'recommendations': recommendations,
#         'memory_available_gb': memory.available / (1024**3),
#         'cpu_usage_percent': cpu_percent,
#         'optimized_for_hcp': True,
#         'hcp_ready': readiness_score >= 50  # Seuil encore plus bas
#     }


# def train_with_hcp_structure_data(config, training_data, force_full_training=True):
#     """Fonction principale pour entraîner avec TOUTES les données HCP - VERSION CORRIGÉE"""
#     print("🚀 ENTRAÎNEMENT AVEC TOUTES LES DONNÉES HCP")
#     print("="*60)
    
#     # Vérifications initiales
#     if not training_data:
#         print("❌ Aucune donnée d'entraînement HCP fournie")
#         return None
    
#     print(f"📊 ANALYSE INITIALE:")
#     print(f"   • Total données disponibles: {len(training_data):,}")
    
#     # Vérifier la présence de la structure HCP
#     hcp_structure_count = sum(1 for item in training_data if item.get('data_source') == 'nouvelle_structure')
#     legacy_structure_count = len(training_data) - hcp_structure_count
    
#     print(f"   • Structure HCP moderne: {hcp_structure_count:,} éléments")
#     print(f"   • Structure legacy: {legacy_structure_count:,} éléments")
    
#     # Analyser la richesse des données HCP
#     territories = set()
#     indicators = set()
#     sources = set()
    
#     for item in training_data[:1000]:  # Échantillon pour analyse rapide
#         territory = item.get('territory', item.get('original_territory'))
#         if territory:
#             territories.add(territory)
        
#         indicator = item.get('variable', item.get('indicateur'))
#         if indicator:
#             indicators.add(indicator)
        
#         source = item.get('source_data', item.get('source'))
#         if source:
#             sources.add(source)
    
#     print(f"   • Territoires détectés: {len(territories)}")
#     print(f"   • Indicateurs détectés: {len(indicators)}")
#     print(f"   • Sources détectées: {len(sources)}")
    
#     # Initialiser le trainer HCP
#     trainer = HCPChatbotTrainer(config)
    
#     # FORCE L'UTILISATION DE TOUTES LES DONNÉES
#     if force_full_training:
#         print(f"\n🎯 MODE ENTRAÎNEMENT COMPLET FORCÉ")
#         print(f"   • TOUTES les {len(training_data):,} données seront utilisées")
#         print(f"   • Aucun sous-ensemble ne sera créé")
        
#         # Vérifications système avec seuil plus permissif
#         system_check = check_system_resources_hcp(min_ram_gb=3.0)
#         if not system_check['hcp_ready']:
#             print("⚠️ Ressources limitées mais entraînement autorisé")
#             print("💡 Conseils: Surveillez la RAM et fermez d'autres applications")
#         else:
#             print("✅ Système prêt pour l'entraînement complet")
        
#         # Recommandation sur la validation
#         use_validation = len(training_data) > 1000
#         print(f"   • Validation croisée: {'Activée' if use_validation else 'Désactivée'}")
        
#         # Lancer l'entraînement HCP avec toutes les données
#         result = trainer.train_model(
#             training_data, 
#             use_validation=use_validation,
#             force_use_all_data=True  # PARAMÈTRE FORCÉ
#         )
        
#         return trainer
    
#     else:
#         print("❌ Mode entraînement partiel non autorisé dans cette version")
#         return None


# if __name__ == "__main__":
#     try:
#         from config import Config
#         from data_processor import HCPDataProcessor
        
#         print("Démarrage de l'entraînement HCP avec TOUTES les données...")
        
#         # Charger les données HCP
#         processor = HCPDataProcessor(Config)
#         data = processor.load_all_data()
        
#         if not data.empty:
#             qa_pairs = processor.create_qa_pairs()
#             print(f"Données HCP chargées: {len(qa_pairs):,} paires QA")
            
#             # Entraînement complet avec toutes les données
#             trainer = train_with_hcp_structure_data(
#                 Config, 
#                 qa_pairs, 
#                 force_full_training=True  # FORCE L'UTILISATION COMPLÈTE
#             )
            
#             if trainer:
#                 print("✅ Entraînement terminé avec succès sur toutes les données!")
#             else:
#                 print("❌ Erreur lors de l'entraînement")
                
#         else:
#             print("❌ Aucune donnée HCP chargée")
            
#     except ImportError:
#         print("❌ Modules requis non disponibles. Assurez-vous que config.py et data_processor.py existent.")






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