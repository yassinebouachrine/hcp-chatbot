import os

class Config:
    # Chemins des données HCP
    DATA_PATHS = {
        "hcp_qa_pairs": "data/indicators.json"
    }
    DATA_PATH = "data/indicators.json"  # Gardé pour compatibilité
    MODEL_PATH = "models/models_hcp"
    BASE_MODEL = "distilgpt2"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Configuration d'entraînement optimisée pour éviter l'erreur CUDA
    NUM_EPOCHS = 3  # Réduit pour éviter le surapprentissage sur 39k échantillons
    BATCH_SIZE = 64  # Taille par device (ajuster si OOM)
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 80  # Réduit pour réduire l'utilisation mémoire
    GRADIENT_ACCUMULATION_STEPS = 4  # Augmenté pour compenser le batch size réduit
    SIMILARITY_THRESHOLD = 0.75
    MAX_RESPONSE_LENGTH = 200
    TEMPERATURE = 0.3
    DATALOADER_NUM_WORKERS = 0  # IMPORTANT: 0 pour éviter les erreurs CUDA multiprocessing
    WEIGHT_DECAY = 0.01

    # Configuration système optimisée
    USE_CUDA = True
    # NOTE: Désactivé par défaut pour éviter les problèmes fp16+checkpointing. Activer manuellement
    # seulement si vous avez vérifié la compatibilité (transformers/accelerate/torch versions stables).
    FP16 = False
    HF_TOKEN = os.getenv('HUGGING_FACE_TOKEN', None)
    LOG_LEVEL = "INFO"

    # Contrôles additionnels pour éviter l'erreur "Attempting to unscale FP16 gradients."
    # - GRADIENT_CHECKPOINTING : active/désactive l'utilisation du gradient checkpointing
    # - FORCE_DISABLE_FP16_IF_INCOMPATIBLE : si True, le système forcera fp16=False si on détecte
    #   certaines incompatibilités runtime. (Le trainer peut aussi gérer une retentative sans fp16.)
    GRADIENT_CHECKPOINTING = True
    FORCE_DISABLE_FP16_IF_INCOMPATIBLE = True

    # Split train/validation
    TRAIN_RATIO = 0.7
    VALIDATION_RATIO = 0.3

    # Configuration pour éviter les problèmes multiprocessing
    TOKENIZERS_PARALLELISM = False
    CUDA_MULTIPROCESSING = "spawn"

    # Historique et interface
    SAVE_CONVERSATION_HISTORY = True
    CONVERSATION_HISTORY_PATH = "data/conversation_history.json"
    FLASK_HOST = "0.0.0.0"
    FLASK_PORT = 5000
    FLASK_DEBUG = True

    # Réponses par défaut adaptées aux données HCP
    DEFAULT_RESPONSES = [
        "Je ne trouve pas d'informations précises sur cette question dans ma base de données HCP. Pouvez-vous reformuler en précisant le territoire et l'indicateur démographique recherché ?",
        "Cette information spécifique n'est pas disponible dans mes données HCP. Je peux vous aider avec la population légale, municipale, les tranches d'âge, ou les indicateurs matrimoniaux pour différents territoires du Maroc.",
        "Je n'ai pas de données exactes correspondant à votre question. Essayez de mentionner un territoire précis (ex: 'Ensemble du territoire national') et un indicateur spécifique HCP.",
        "Désolé, cette statistique n'est pas dans ma base HCP. Demandez-moi par exemple la population légale d'une région ou le pourcentage d'une tranche d'âge."
    ]

    GREETING_RESPONSES = [
        "Bonjour ! Je suis l'assistant spécialisé dans les statistiques du HCP Maroc. Je dispose de données démographiques complètes : population légale, municipale, tranches d'âge, indicateurs matrimoniaux et plus. Posez-moi une question précise !",
        "Salut ! J'ai accès aux dernières données HCP du Maroc avec plus de 140 000 statistiques. Demandez-moi par exemple : 'Quelle est la population légale du Maroc ?' ou 'Pourcentage de 0-4 ans au niveau national ?'",
        "Bienvenue ! Je suis votre expert en statistiques démographiques HCP. Je connais la population, les ménages, l'éducation et l'emploi par territoire. Comment puis-je vous aider avec les données officielles ?",
        "Bonjour ! Assistant HCP à votre service. Je dispose de statistiques détaillées sur 2000+ territoires marocains. Spécifiez votre territoire et l'indicateur souhaité pour des données précises."
    ]

    # Configuration spécifique pour la nouvelle structure HCP
    NEW_DATA_STRUCTURE = {
        "root_key": "qa_pairs",
        "question_key": "question",
        "answer_key": "answer",
        "territory_key": "territoire",
        "indicator_key": "indicateur",
        "gender_key": "genre",
        "source_key": "source"
    }

    # Mappage des indicateurs HCP
    HCP_INDICATOR_MAPPING = {
        "population_legale": "Population légale",
        "population_municipale": "Population municipale",
        "pourcentage_masculin": "Pourcentage population masculine",
        "pourcentage_feminin": "Pourcentage population féminine",
        "matrimonial_celibataire": "Pourcentage célibataires",
        "matrimonial_marie": "Pourcentage mariés",
        "matrimonial_divorce": "Pourcentage divorcés",
        "matrimonial_veuf": "Pourcentage veufs",
        "age_0_4": "Population 0-4 ans",
        "age_5_9": "Population 5-9 ans",
        "age_10_14": "Population 10-14 ans",
        "emploi_actif": "Population active",
        "emploi_chomage": "Taux de chômage",
        "education_scolarisation": "Taux de scolarisation",
        "menage_taille": "Taille moyenne des ménages",
        "logement_type": "Type de logement"
    }

    # Sources de données HCP reconnues
    HCP_DATA_SOURCES = {
        "population": "Données de population HCP",
        "menages": "Données des ménages HCP",
        "emploi": "Données d'emploi HCP",
        "education": "Données d'éducation HCP",
        "logement": "Données de logement HCP",
        "general": "Données générales HCP"
    }

    # Configuration des tokens spéciaux HCP
    HCP_SPECIAL_TOKENS = {
        "context_start": "<|hcp|>",
        "territory": "<|territory|>",
        "indicator": "<|indicator|>",
        "gender": "<|genre|>",
        "source": "<|source|>",
        "user": "<|user|>",
        "assistant": "<|assistant|>",
        "end": "<|endoftext|>"
    }

    @classmethod
    def get_all_data_files(cls):
        """Retourne tous les fichiers de données configurés"""
        return list(cls.DATA_PATHS.values())

    @classmethod
    def get_data_file_by_type(cls, data_type):
        """Retourne le fichier de données par type"""
        return cls.DATA_PATHS.get(data_type.lower())

    @classmethod
    def validate_config(cls):
        """Valide la configuration et vérifie les fichiers de données HCP"""
        # Créer les dossiers nécessaires
        os.makedirs("data", exist_ok=True)
        os.makedirs(cls.MODEL_PATH, exist_ok=True)
        os.makedirs(os.path.dirname(cls.CONVERSATION_HISTORY_PATH), exist_ok=True)

        missing_files = []
        existing_files = []

        print("Validation de la configuration HCP...")

        # Vérifier les fichiers de données
        for data_type, file_path in cls.DATA_PATHS.items():
            if os.path.exists(file_path):
                existing_files.append(f"{data_type}: {file_path}")
                try:
                    import json
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if cls.NEW_DATA_STRUCTURE["root_key"] in data:
                        qa_count = len(data[cls.NEW_DATA_STRUCTURE["root_key"]])
                        metadata = data.get("metadata", {})
                        total_qa = metadata.get("total_qa_pairs", qa_count)
                        territories = metadata.get("unique_territories", "unknown")

                        print(f"Structure HCP moderne détectée:")
                        print(f"   • {total_qa:,} paires QA")
                        print(f"   • {territories} territoires uniques" if territories != "unknown" else f"   • {qa_count:,} paires QA chargées")

                        if "sources_stats" in metadata:
                            print(f"   • Sources: {', '.join(metadata['sources_stats'].keys())}")

                        sample_items = data[cls.NEW_DATA_STRUCTURE["root_key"]][:3]
                        required_fields = ["question", "answer", "territoire", "indicateur"]

                        valid_samples = 0
                        for item in sample_items:
                            if all(field in item for field in required_fields):
                                valid_samples += 1

                        print(f"   • Échantillons valides: {valid_samples}/{len(sample_items)}")

                    else:
                        print(f"Structure legacy détectée dans {file_path}")

                except Exception as e:
                    print(f"Erreur lors de la validation de {file_path}: {e}")
            else:
                missing_files.append(f"{data_type}: {file_path}")

        if existing_files:
            print("Fichiers de données HCP trouvés:")
            for file_info in existing_files:
                print(f"  - {file_info}")

        if missing_files:
            print("Fichiers de données HCP manquants:")
            for file_info in missing_files:
                print(f"  - {file_info}")

        # Fallback vers fichier unique
        if not existing_files and hasattr(cls, 'DATA_PATH') and os.path.exists(cls.DATA_PATH):
            print(f"Fichier de données unique trouvé: {cls.DATA_PATH}")
            existing_files.append(f"legacy: {cls.DATA_PATH}")
        elif not existing_files:
            print("Aucun fichier de données HCP trouvé!")
            print("Assurez-vous que vos fichiers JSON avec structure HCP sont présents dans le dossier data/")

        return len(existing_files) > 0

    @classmethod
    def get_optimized_training_params(cls):
        """Retourne les paramètres optimisés pour l'entraînement sans erreur CUDA"""
        return {
            "batch_size": cls.BATCH_SIZE,
            "max_length": cls.MAX_LENGTH,
            "gradient_accumulation_steps": cls.GRADIENT_ACCUMULATION_STEPS,
            "dataloader_num_workers": cls.DATALOADER_NUM_WORKERS,
            "fp16": cls.FP16,
            "tokenizers_parallelism": cls.TOKENIZERS_PARALLELISM,
            "effective_batch_size": cls.BATCH_SIZE * cls.GRADIENT_ACCUMULATION_STEPS,
            "gradient_checkpointing": cls.GRADIENT_CHECKPOINTING
        }

    @classmethod
    def setup_multiprocessing_env(cls):
        """Configure l'environnement pour éviter les erreurs CUDA multiprocessing"""
        import torch

        # Variables d'environnement
        os.environ['TOKENIZERS_PARALLELISM'] = str(cls.TOKENIZERS_PARALLELISM).lower()

        # Configuration PyTorch multiprocessing
        if torch.cuda.is_available() and hasattr(torch.multiprocessing, 'set_start_method'):
            try:
                torch.multiprocessing.set_start_method(cls.CUDA_MULTIPROCESSING, force=True)
                print(f"Méthode multiprocessing configurée: {cls.CUDA_MULTIPROCESSING}")
            except RuntimeError as e:
                print(f"Avertissement: {e}")

        return True

    @classmethod
    def get_memory_optimized_config(cls, available_memory_gb=None):
        """Retourne une configuration optimisée selon la mémoire disponible"""
        if available_memory_gb is None:
            try:
                import psutil
                available_memory_gb = psutil.virtual_memory().available / (1024**3)
            except:
                available_memory_gb = 8.0

        # Configuration adaptative selon la mémoire
        if available_memory_gb < 8:
            return {
                "batch_size": 32,
                "max_length": 64,
                "gradient_accumulation_steps": 8,
                "fp16": True,
                "recommendation": "Configuration mémoire faible"
            }
        elif available_memory_gb < 16:
            return {
                "batch_size": 64,
                "max_length": 80,
                "gradient_accumulation_steps": 4,
                "fp16": True,
                "recommendation": "Configuration mémoire standard"
            }
        else:
            return {
                "batch_size": 128,
                "max_length": 96,
                "gradient_accumulation_steps": 2,
                "fp16": True,
                "recommendation": "Configuration haute mémoire"
            }

    @classmethod
    def normalize_runtime_flags(cls):
        """Ajuste et retourne les flags runtime pour éviter incompatibilités connues.

        - Désactive FP16 si CUDA absent ou si l'utilisateur a explicitement demandé de forcer
          la désactivation.
        - Si FP16 est activé conjointement avec GRADIENT_CHECKPOINTING, on affiche un avertissement
          (le trainer applique aussi une stratégie de retentative sans fp16).
        """
        try:
            import torch
        except Exception:
            torch = None

        recommended = {
            'fp16': cls.FP16,
            'gradient_checkpointing': cls.GRADIENT_CHECKPOINTING
        }

        if torch is None or not (hasattr(torch, 'cuda') and torch.cuda.is_available()):
            if cls.FP16:
                print("⚠️ CUDA non disponible : désactivation automatique de FP16")
            recommended['fp16'] = False

        # Si force disable et combinaison risquée
        if recommended['fp16'] and recommended['gradient_checkpointing'] and cls.FORCE_DISABLE_FP16_IF_INCOMPATIBLE:
            print("⚠️ FP16 + gradient_checkpointing peut causer 'Attempting to unscale FP16 gradients'.")
            print("    Par sécurité, FP16 est désactivé par défaut. Activez manuellement si vous savez que votre stack est compatible.")
            recommended['fp16'] = False

        return recommended


if __name__ == "__main__":
    print("=== Configuration HCP Optimisée ===\n")

    # Setup de l'environnement
    Config.setup_multiprocessing_env()

    # Normalize runtime flags
    runtime_flags = Config.normalize_runtime_flags()
    print("Flags runtime recommandés:")
    for k, v in runtime_flags.items():
        print(f"  {k}: {v}")

    # Validation
    config_valid = Config.validate_config()

    if config_valid:
        print("\n=== Paramètres d'entraînement optimisés ===")
        training_params = Config.get_optimized_training_params()
        for key, value in training_params.items():
            print(f"  {key}: {value}")

        print("\n=== Configuration adaptative mémoire ===")
        memory_config = Config.get_memory_optimized_config()
        for key, value in memory_config.items():
            print(f"  {key}: {value}")

        print("\nConfiguration prête pour l'entraînement sans erreur CUDA!")

    else:
        print("Configuration invalide - vérifiez vos fichiers de données HCP")




