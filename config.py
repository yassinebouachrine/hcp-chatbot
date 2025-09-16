# import os

# class Config:
#     # CORRECTION: Mise à jour des chemins pour la nouvelle structure
#     DATA_PATHS = {
#         "qa_pairs": "data/indicators.json"  # Nouveau format avec qa_pairs dans le JSON
#     }
#     DATA_PATH = "data/indicators.json"  # Gardé pour compatibilité
#     MODEL_PATH = "models/models_hcp"
#     BASE_MODEL = "distilgpt2"
#     EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#     NUM_EPOCHS = 3
#     BATCH_SIZE = 256
#     LEARNING_RATE = 2e-5
#     MAX_LENGTH = 64
#     GRADIENT_ACCUMULATION_STEPS = 1
#     SIMILARITY_THRESHOLD = 0.85
#     MAX_RESPONSE_LENGTH = 150
#     TEMPERATURE = 0.3
#     DATALOADER_NUM_WORKERS = 8
#     USE_CUDA = True
#     FP16 = True
#     HF_TOKEN = os.getenv('HUGGING_FACE_TOKEN', None)
#     LOG_LEVEL = "INFO"
#     SAVE_CONVERSATION_HISTORY = True
#     CONVERSATION_HISTORY_PATH = "data/conversation_history.json"
#     FLASK_HOST = "0.0.0.0"
#     FLASK_PORT = 5000
#     FLASK_DEBUG = True
    
#     DEFAULT_RESPONSES = [
#         "Je ne trouve pas d'informations précises sur cette question dans ma base de données HCP. Pouvez-vous reformuler en précisant le territoire et le type de statistique recherché ?",
#         "Cette information spécifique n'est pas disponible dans mes données. Je peux vous aider avec la population légale, municipale, ou les tranches d'âge pour différents territoires du Maroc.",
#         "Je n'ai pas de données exactes correspondant à votre question. Essayez de mentionner un territoire précis (ex: 'Ensemble du territoire national') et un indicateur spécifique."
#     ]
    
#     GREETING_RESPONSES = [
#         "Bonjour ! Je suis l'assistant statistique du HCP. Je peux vous fournir des données précises sur la population légale, municipale et les tranches d'âge au Maroc. Posez-moi une question spécifique !",
#         "Salut ! Je dispose de statistiques démographiques détaillées du Maroc. Demandez-moi par exemple : 'Quelle est la population légale du Maroc ?' ou 'Quel est le pourcentage de 0-4 ans au niveau national ?'",
#         "Bienvenue ! Je suis spécialisé dans les données HCP du Maroc. Je connais la population légale, municipale et la répartition par âge. Comment puis-je vous aider ?"
#     ]
    
#     # NOUVEAU: Configuration spécifique pour la nouvelle structure
#     NEW_DATA_STRUCTURE = {
#         "root_key": "qa_pairs",  # Clé racine dans le JSON
#         "question_key": "question",
#         "answer_key": "answer", 
#         "territory_key": "territoire",
#         "indicator_key": "indicateur", 
#         "gender_key": "genre"
#     }
    
#     @classmethod
#     def get_all_data_files(cls):
#         return list(cls.DATA_PATHS.values())
    
#     @classmethod
#     def get_data_file_by_type(cls, data_type):
#         return cls.DATA_PATHS.get(data_type.lower())
    
#     @classmethod
#     def validate_config(cls):
#         os.makedirs("data", exist_ok=True)
#         os.makedirs(cls.MODEL_PATH, exist_ok=True)
#         os.makedirs(os.path.dirname(cls.CONVERSATION_HISTORY_PATH), exist_ok=True)
        
#         missing_files = []
#         existing_files = []
        
#         for data_type, file_path in cls.DATA_PATHS.items():
#             if os.path.exists(file_path):
#                 existing_files.append(f"{data_type}: {file_path}")
#                 # NOUVEAU: Vérifier la structure du fichier
#                 try:
#                     import json
#                     with open(file_path, 'r', encoding='utf-8') as f:
#                         data = json.load(f)
                    
#                     # Vérifier si la nouvelle structure est présente
#                     if cls.NEW_DATA_STRUCTURE["root_key"] in data:
#                         qa_count = len(data[cls.NEW_DATA_STRUCTURE["root_key"]])
#                         print(f"✅ Structure moderne détectée: {qa_count} paires QA")
#                     else:
#                         print(f"ℹ️ Structure legacy détectée dans {file_path}")
                        
#                 except Exception as e:
#                     print(f"⚠️ Erreur lors de la validation de {file_path}: {e}")
#             else:
#                 missing_files.append(f"{data_type}: {file_path}")
        
#         if existing_files:
#             print("✅ Fichiers de données trouvés:")
#             for file_info in existing_files:
#                 print(f"  - {file_info}")
        
#         if missing_files:
#             print("⚠️ Fichiers de données manquants:")
#             for file_info in missing_files:
#                 print(f"  - {file_info}")
        
#         if not existing_files and hasattr(cls, 'DATA_PATH') and os.path.exists(cls.DATA_PATH):
#             print(f"ℹ️ Fichier de données unique trouvé: {cls.DATA_PATH}")
#             existing_files.append(f"legacy: {cls.DATA_PATH}")
#         elif not existing_files:
#             print("❌ Aucun fichier de données trouvé!")
#             print("Assurez-vous que vos fichiers JSON sont présents dans le dossier data/")
        
#         # Validations des paramètres
#         assert cls.BATCH_SIZE >= 1, "BATCH_SIZE doit être >= 1"
#         assert cls.NUM_EPOCHS >= 1, "NUM_EPOCHS doit être >= 1"
#         assert cls.LEARNING_RATE > 0, "LEARNING_RATE doit être > 0"
#         assert 0 <= cls.SIMILARITY_THRESHOLD <= 1, "SIMILARITY_THRESHOLD doit être entre 0 et 1"
        
#         print("✅ Configuration validée avec succès")
#         return len(existing_files) > 0
    
#     @classmethod
#     def get_training_summary(cls):
#         return {
#             "Modèle de base": cls.BASE_MODEL,
#             "Modèle d'embedding": cls.EMBEDDING_MODEL,
#             "Nombre d'époques": cls.NUM_EPOCHS,
#             "Taille de batch": cls.BATCH_SIZE,
#             "Taux d'apprentissage": cls.LEARNING_RATE,
#             "Longueur maximale": cls.MAX_LENGTH,
#             "Accumulation de gradients": cls.GRADIENT_ACCUMULATION_STEPS,
#             "Seuil de similarité": cls.SIMILARITY_THRESHOLD,
#             "Température": cls.TEMPERATURE,
#             "Structure des données": "Moderne (qa_pairs)" if hasattr(cls, 'NEW_DATA_STRUCTURE') else "Legacy"
#         }
    
#     @classmethod
#     def get_chatbot_summary(cls):
#         return {
#             "Seuil de similarité": cls.SIMILARITY_THRESHOLD,
#             "Longueur max réponse": cls.MAX_RESPONSE_LENGTH,
#             "Température": cls.TEMPERATURE,
#             "Sauvegarde historique": cls.SAVE_CONVERSATION_HISTORY,
#             "Fichiers de données": list(cls.DATA_PATHS.keys()),
#             "Structure supportée": "qa_pairs moderne"
#         }

# if __name__ == "__main__":
#     print("=== Test de la configuration HCP pour nouvelle structure ===\n")
#     config_valid = Config.validate_config()
    
#     if config_valid:
#         print("\n=== Configuration d'entraînement ===")
#         training_config = Config.get_training_summary()
#         for key, value in training_config.items():
#             print(f"  {key}: {value}")
#     else:
#         print("❌ Configuration invalide - vérifiez vos fichiers de données")








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













# import os
# from typing import Dict, List, Any, Optional
# from pathlib import Path

# class Config:
#     """Configuration optimisée pour le système HCP Chatbot avancé."""
    
#     # =========================================
#     # CONFIGURATION DES DONNÉES
#     # =========================================
    
#     # Chemins des données (compatible avec la nouvelle architecture)
#     DATA_PATHS = {
#         "hcp_qa_pairs": "data/indicators.json",
#         "population_data": "data/population.json",
#         "menages_data": "data/menages.json",
#         "territories_mapping": "data/territories.json",
#         "indicators_mapping": "data/indicators_mapping.json"
#     }
    
#     # Compatibilité avec l'ancienne structure
#     DATA_PATH = "data/indicators.json"
    
#     # Configuration de la structure des données
#     NEW_DATA_STRUCTURE = {
#         "root_key": "qa_pairs",
#         "question_key": "question",
#         "answer_key": "answer", 
#         "territory_key": "territoire",
#         "indicator_key": "indicateur", 
#         "gender_key": "genre",
#         "source_key": "source"
#     }
    
#     # =========================================
#     # CONFIGURATION DES MODÈLES
#     # =========================================
    
#     MODEL_PATH = "models/models_hcp"
#     BASE_MODEL = "distilgpt2"
#     EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
#     # Modèles alternatifs pour recherche multilingue
#     MULTILINGUAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#     FRENCH_EMBEDDING_MODEL = "sentence-transformers/distiluse-base-multilingual-cased"
    
#     # =========================================
#     # CONFIGURATION DE L'ALGORITHME DE RECHERCHE
#     # =========================================
    
#     # Seuils de similarité pour les différentes méthodes
#     SIMILARITY_THRESHOLD = 0.65  # Seuil pour recherche sémantique
#     FUZZY_THRESHOLD = 0.60       # Seuil pour recherche fuzzy
#     TFIDF_THRESHOLD = 0.15       # Seuil pour recherche TF-IDF
#     FIELD_MATCH_THRESHOLD = 0.3  # Seuil pour correspondance par champs
    
#     # Configuration de la recherche multi-niveaux
#     SEARCH_CONFIG = {
#         'use_semantic_search': True,
#         'use_tfidf_search': True,
#         'use_field_specific_search': True,
#         'use_fuzzy_search': True,
#         'max_results_per_method': 10,
#         'final_results_limit': 10,
#         'enable_result_fusion': True,
#         'enable_context_boosting': True
#     }
    
#     # Configuration des poids pour la fusion des résultats
#     SEARCH_WEIGHTS = {
#         'semantic': 0.35,
#         'tfidf': 0.25,
#         'field_specific': 0.25,
#         'fuzzy': 0.15
#     }
    
#     # Configuration TF-IDF
#     TFIDF_CONFIG = {
#         'max_features': 5000,
#         'ngram_range': (1, 3),
#         'min_df': 2,
#         'max_df': 0.95,
#         'use_idf': True,
#         'smooth_idf': True
#     }
    
#     # =========================================
#     # CONFIGURATION DU MATCHING TERRITORIAL
#     # =========================================
    
#     # Aliases territoriaux pour améliorer la recherche
#     TERRITORY_ALIASES = {
#         "maroc": ["royaume du maroc", "territoire national", "ensemble du territoire national"],
#         "casa": ["casablanca", "grand casablanca"],
#         "rabat": ["rabat sale", "rabat-sale", "capitale"],
#         "agadir": ["agadir ida outanane", "agadir ida-outanane"],
#         "fes": ["fès", "fez"],
#         "marrakech": ["marrakesh"],
#         "tanger": ["tangier", "tanger-tetouan"],
#         "oujda": ["oujda angad"]
#     }
    
#     # Préfixes territoriaux à ignorer lors de la normalisation
#     TERRITORY_PREFIXES = [
#         "commune", "municipalite", "ville", "province", 
#         "prefecture", "region", "cercle", "arrondissement",
#         "ca", "municipal", "territorial", "ensemble du"
#     ]
    
#     # Configuration du matcher territorial
#     TERRITORY_MATCHING = {
#         'use_fuzzy_matching': True,
#         'fuzzy_threshold': 0.6,
#         'enable_token_matching': True,
#         'min_token_length': 3,
#         'use_aliases': True,
#         'normalize_prefixes': True
#     }
    
#     # =========================================
#     # CONFIGURATION DES INDICATEURS
#     # =========================================
    
#     # Mapping complet des indicateurs HCP organisé par domaines
#     HCP_INDICATOR_DOMAINS = {
#         'population': {
#             'demographie_base': {
#                 'population_legale': ['population legale', 'pop legale', 'habitants legaux'],
#                 'population_municipale': ['population municipale', 'pop municipale', 'habitants municipaux'],
#                 'densite': ['densite', 'densite population', 'hab/km2']
#             },
#             'structure_sexe': {
#                 'pourcentage_masculin': ['masculin', 'hommes', 'sexe masculin', '% masculin'],
#                 'pourcentage_feminin': ['feminin', 'femmes', 'sexe feminin', '% feminin']
#             },
#             'structure_age': {
#                 'age_0_4': ['0-4 ans', '0 4 ans', 'moins de 5 ans'],
#                 'age_5_9': ['5-9 ans', '5 9 ans'],
#                 'age_10_14': ['10-14 ans', '10 14 ans'],
#                 'age_15_19': ['15-19 ans', '15 19 ans'],
#                 'age_20_24': ['20-24 ans', '20 24 ans'],
#                 'age_25_29': ['25-29 ans', '25 29 ans'],
#                 'age_30_34': ['30-34 ans', '30 34 ans'],
#                 'age_35_39': ['35-39 ans', '35 39 ans'],
#                 'age_40_44': ['40-44 ans', '40 44 ans'],
#                 'age_45_49': ['45-49 ans', '45 49 ans'],
#                 'age_50_54': ['50-54 ans', '50 54 ans'],
#                 'age_55_59': ['55-59 ans', '55 59 ans'],
#                 'age_60_64': ['60-64 ans', '60 64 ans'],
#                 'age_65_69': ['65-69 ans', '65 69 ans'],
#                 'age_70_74': ['70-74 ans', '70 74 ans'],
#                 'age_75_plus': ['75 ans ou plus', '75 ans et plus', 'plus de 75 ans']
#             },
#             'etat_matrimonial': {
#                 'celibataire': ['celibataire', 'non marie'],
#                 'marie': ['marie', 'matrimonial marie'],
#                 'divorce': ['divorce', 'divorce'],
#                 'veuf': ['veuf', 'veuve']
#             },
#             'education': {
#                 'analphabetisme_10_plus': ['analphabetisme 10 ans', 'taux analphabetisme 10+'],
#                 'analphabetisme_15_plus': ['analphabetisme 15 ans', 'taux analphabetisme 15+'],
#                 'scolarisation_6_11': ['scolarisation 6-11', 'taux scolarisation 6 11'],
#                 'niveau_primaire': ['primaire', 'enseignement primaire'],
#                 'niveau_secondaire': ['secondaire', 'collegial', 'qualifiant'],
#                 'niveau_superieur': ['superieur', 'universitaire', 'enseignement superieur']
#             },
#             'langues': {
#                 'arabe': ['arabe', 'langue arabe'],
#                 'amazigh': ['amazigh', 'tifinagh', 'berbere'],
#                 'francais': ['francais', 'langue francaise'],
#                 'anglais': ['anglais', 'langue anglaise'],
#                 'langues_locales': ['darija', 'tachelhit', 'tamazight', 'tarifit', 'hassania']
#             },
#             'emploi': {
#                 'population_active': ['population active', 'actifs'],
#                 'taux_activite': ['taux activite', 'taux d\'activite'],
#                 'taux_chomage': ['taux chomage', 'chomage', 'taux de chomage'],
#                 'employes': ['employes', 'salaries'],
#                 'independants': ['independant', 'travailleur independant'],
#                 'employeurs': ['employeur', 'chef entreprise']
#             },
#             'handicap': {
#                 'taux_handicap': ['taux handicap', 'prevalence handicap', 'handicap']
#             }
#         },
#         'menages': {
#             'structure': {
#                 'nombre_menages': ['nombre menages', 'menages'],
#                 'taille_moyenne': ['taille moyenne', 'taille menage'],
#                 'menages_sedentaires': ['menages sedentaires', 'sedentaires']
#             },
#             'logement': {
#                 'villa': ['villa', 'etage villa'],
#                 'appartement': ['appartement'],
#                 'maison_marocaine': ['maison marocaine'],
#                 'maison_sommaire': ['maison sommaire', 'bidonville'],
#                 'logement_rural': ['logement rural']
#             },
#             'confort': {
#                 'eau_courante': ['eau courante', 'acces eau'],
#                 'electricite': ['electricite', 'acces electricite'],
#                 'cuisine': ['cuisine', 'piece cuisine'],
#                 'toilettes': ['toilettes', 'wc', 'w.c.'],
#                 'piece_eau': ['piece eau', 'salle de bain']
#             },
#             'statut_occupation': {
#                 'proprietaire': ['proprietaire'],
#                 'locataire': ['locataire']
#             },
#             'age_logement': {
#                 'moins_10_ans': ['moins 10 ans', 'neuf'],
#                 '10_19_ans': ['10-19 ans'],
#                 '20_49_ans': ['20-49 ans'],
#                 '50_ans_plus': ['50 ans plus', 'ancien']
#             },
#             'services': {
#                 'assainissement': ['assainissement', 'reseau assainissement'],
#                 'fosse_septique': ['fosse septique'],
#                 'evacuation_dechets': ['evacuation dechets', 'ordures'],
#                 'camion_commune': ['camion commune', 'collecte ordures']
#             },
#             'combustible': {
#                 'gaz': ['gaz', 'gaz cuisson'],
#                 'electricite_cuisson': ['electricite cuisson'],
#                 'charbon': ['charbon'],
#                 'bois': ['bois', 'bois energie']
#             },
#             'accessibilite': {
#                 'distance_route': ['distance route', 'route goudronnee']
#             }
#         }
#     }
    
#     # =========================================
#     # CONFIGURATION DE L'ENTRAÎNEMENT
#     # =========================================
    
#     NUM_EPOCHS = 3
#     BATCH_SIZE = 128
#     LEARNING_RATE = 2e-5
#     MAX_LENGTH = 128
#     MAX_RESPONSE_LENGTH = 300
#     GRADIENT_ACCUMULATION_STEPS = 2
#     WEIGHT_DECAY = 0.01
#     TEMPERATURE = 0.3
#     DATALOADER_NUM_WORKERS = 6
    
#     # Configuration avancée pour l'entraînement
#     TRAINING_CONFIG = {
#         'use_gradient_checkpointing': True,
#         'fp16': True,
#         'dataloader_pin_memory': True,
#         'remove_unused_columns': False,
#         'save_strategy': 'epoch',
#         'evaluation_strategy': 'epoch',
#         'logging_steps': 100,
#         'warmup_steps': 500,
#         'save_total_limit': 3
#     }
    
#     # =========================================
#     # CONFIGURATION FLASK
#     # =========================================
    
#     FLASK_HOST = "0.0.0.0"
#     FLASK_PORT = 5000
#     FLASK_DEBUG = True
#     SECRET_KEY = os.getenv('SECRET_KEY', 'hcp-chatbot-secret-key-2024')
#     MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2MB
    
#     # Configuration des CORS (si nécessaire)
#     CORS_CONFIG = {
#         'origins': ['http://localhost:3000', 'http://127.0.0.1:3000'],
#         'methods': ['GET', 'POST', 'OPTIONS'],
#         'allow_headers': ['Content-Type', 'Authorization']
#     }
    
#     # =========================================
#     # CONFIGURATION DU CACHE ET PERFORMANCE
#     # =========================================
    
#     # Cache des embeddings
#     EMBEDDINGS_CACHE_PATH = "data/embeddings_cache.pkl"
#     EMBEDDING_BATCH_SIZE = 32
#     ENABLE_EMBEDDING_CACHE = True
    
#     # Cache de recherche
#     SEARCH_CACHE_SIZE = 1000
#     ENABLE_SEARCH_CACHE = True
    
#     # Optimisations mémoire
#     MEMORY_CONFIG = {
#         'max_vocabulary_size': 50000,
#         'max_search_index_entries': 100000,
#         'cleanup_interval_seconds': 3600,
#         'max_conversation_history': 10000
#     }
    
#     # =========================================
#     # CONFIGURATION DES MÉTRIQUES
#     # =========================================
    
#     # Historique des conversations
#     SAVE_CONVERSATION_HISTORY = True
#     CONVERSATION_HISTORY_PATH = "data/conversation_history.json"
#     MAX_HISTORY_SIZE = 1000
    
#     # Métriques avancées
#     ENABLE_ADVANCED_METRICS = True
#     METRICS_CONFIG = {
#         'track_search_performance': True,
#         'track_territory_detection': True,
#         'track_indicator_detection': True,
#         'track_confidence_scores': True,
#         'enable_performance_alerts': True,
#         'alert_response_time_threshold': 5.0,
#         'alert_error_rate_threshold': 20.0
#     }
    
#     # =========================================
#     # CONFIGURATION DU LOGGING
#     # =========================================
    
#     LOG_LEVEL = "INFO"
#     LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     LOG_FILE = "logs/hcp_chatbot.log"
    
#     # Configuration logging avancée
#     LOGGING_CONFIG = {
#         'version': 1,
#         'disable_existing_loggers': False,
#         'formatters': {
#             'detailed': {
#                 'format': '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
#             },
#             'simple': {
#                 'format': '%(levelname)s - %(message)s'
#             }
#         },
#         'handlers': {
#             'console': {
#                 'class': 'logging.StreamHandler',
#                 'level': 'INFO',
#                 'formatter': 'simple'
#             },
#             'file': {
#                 'class': 'logging.FileHandler',
#                 'filename': 'logs/hcp_chatbot.log',
#                 'level': 'DEBUG',
#                 'formatter': 'detailed'
#             }
#         },
#         'loggers': {
#             'HCPChatbotOptimized': {
#                 'handlers': ['console', 'file'],
#                 'level': 'DEBUG',
#                 'propagate': False
#             },
#             'FlaskAppOptimized': {
#                 'handlers': ['console', 'file'],
#                 'level': 'INFO',
#                 'propagate': False
#             }
#         },
#         'root': {
#             'level': 'WARNING',
#             'handlers': ['console']
#         }
#     }
    
#     # =========================================
#     # CONFIGURATION SYSTÈME
#     # =========================================
    
#     USE_CUDA = True
#     FP16 = True
#     HF_TOKEN = os.getenv('HUGGING_FACE_TOKEN', None)
    
#     # Configuration des ressources système
#     SYSTEM_CONFIG = {
#         'max_cpu_workers': min(8, os.cpu_count() or 4),
#         'enable_multiprocessing': True,
#         'memory_limit_gb': 8,
#         'gpu_memory_fraction': 0.8 if USE_CUDA else 0
#     }
    
#     # =========================================
#     # RÉPONSES PAR DÉFAUT ET MESSAGES
#     # =========================================
    
#     DEFAULT_RESPONSES = [
#         "Je ne trouve pas d'informations précises sur cette question dans ma base de données HCP. Pouvez-vous reformuler en précisant le territoire et l'indicateur démographique recherché ?",
#         "Cette information spécifique n'est pas disponible dans mes données HCP. Je peux vous aider avec la population légale, municipale, les tranches d'âge, ou les indicateurs matrimoniaux pour différents territoires du Maroc.",
#         "Je n'ai pas de données exactes correspondant à votre question. Essayez de mentionner un territoire précis (ex: 'Ensemble du territoire national') et un indicateur spécifique HCP.",
#         "Désolé, cette statistique n'est pas dans ma base HCP. Demandez-moi par exemple la population légale d'une région ou le pourcentage d'une tranche d'âge."
#     ]
    
#     GREETING_RESPONSES = [
#         "Bonjour ! Je suis l'assistant spécialisé dans les statistiques du HCP Maroc. Je dispose de données démographiques complètes : population légale, municipale, tranches d'âge, indicateurs matrimoniaux et plus. Posez-moi une question précise !",
#         "Salut ! J'ai accès aux dernières données HCP du Maroc avec plus de 140 000 statistiques. Demandez-moi par exemple : 'Quelle est la population légale du Maroc ?' ou 'Pourcentage de 0-4 ans au niveau national ?'",
#         "Bienvenue ! Je suis votre expert en statistiques démographiques HCP. Je connais la population, les ménages, l'éducation et l'emploi par territoire. Comment puis-je vous aider avec les données officielles ?",
#         "Bonjour ! Assistant HCP à votre service. Je dispose de statistiques détaillées sur 2000+ territoires marocains. Spécifiez votre territoire et l'indicateur souhaité pour des données précises."
#     ]
    
#     # Messages d'aide contextuels
#     HELP_MESSAGES = {
#         'territory_missing': "Conseil: Précisez le territoire (région, province, commune) pour des réponses plus précises.",
#         'indicator_missing': "Conseil: Spécifiez l'indicateur recherché (population, âge, éducation, emploi, etc.).",
#         'general_help': "Exemples: 'Population légale de Casablanca', 'Taux de scolarisation 6-11 ans au niveau national', 'Pourcentage de femmes mariées à Agadir'",
#         'data_sources': "Je dispose de données HCP sur la population, les ménages, l'éducation, l'emploi et le logement pour tous les territoires du Maroc."
#     }
    
#     # =========================================
#     # TOKENS SPÉCIAUX ET FORMATAGE
#     # =========================================
    
#     HCP_SPECIAL_TOKENS = {
#         "context_start": "<|hcp|>",
#         "territory": "<|territory|>",
#         "indicator": "<|indicator|>", 
#         "gender": "<|genre|>",
#         "source": "<|source|>",
#         "user": "<|user|>",
#         "assistant": "<|assistant|>",
#         "end": "<|endoftext|>",
#         "confidence": "<|confidence|>",
#         "method": "<|method|>"
#     }
    
#     # Configuration du formatage des réponses
#     RESPONSE_FORMAT = {
#         'include_source': False,
#         'include_confidence': False,
#         'include_territory_context': True,
#         'max_decimal_places': 1,
#         'use_thousands_separator': True,
#         'confidence_threshold_for_warning': 0.6
#     }
    
#     # =========================================
#     # SOURCES DE DONNÉES HCP
#     # =========================================
    
#     HCP_DATA_SOURCES = {
#         "population": "Données de population HCP - Recensement Général",
#         "menages": "Données des ménages HCP - Enquête Nationale", 
#         "emploi": "Données d'emploi HCP - Enquête Nationale sur l'Emploi",
#         "education": "Données d'éducation HCP - Statistiques Éducatives",
#         "logement": "Données de logement HCP - Enquête Logement",
#         "general": "Données générales HCP - Multiple sources"
#     }
    
#     # =========================================
#     # FAQ STATIQUE
#     # =========================================
    
#     STATIC_FAQ = [
#         {
#             'question': 'Qu\'est-ce que le HCP ?',
#             'answer': 'Le HCP (Haut-Commissariat au Plan) est l\'organisme officiel marocain chargé de la production des statistiques nationales, notamment les recensements de population et d\'habitat.',
#             'territory': 'Maroc',
#             'variable': 'info_generale',
#             'source': 'general'
#         },
#         {
#             'question': 'Quelle est la différence entre population légale et municipale ?',
#             'answer': 'La population légale comprend tous les résidents habituels, tandis que la population municipale exclut certaines populations spéciales comme les personnes en institutions.',
#             'territory': 'Maroc',
#             'variable': 'definition_population',
#             'source': 'general'
#         },
#         {
#             'question': 'Comment sont calculés les taux démographiques ?',
#             'answer': 'Les taux démographiques HCP sont calculés en pourcentage de la population totale concernée, avec des méthodologies standardisées selon les recommandations internationales.',
#             'territory': 'Maroc',
#             'variable': 'methodologie',
#             'source': 'general'
#         }
#     ]
    
#     # =========================================
#     # MÉTHODES DE VALIDATION ET UTILITAIRES
#     # =========================================
    
#     @classmethod
#     def create_directories(cls):
#         """Crée tous les répertoires nécessaires."""
#         directories = [
#             'data',
#             'logs',
#             os.path.dirname(cls.MODEL_PATH),
#             os.path.dirname(cls.EMBEDDINGS_CACHE_PATH),
#             os.path.dirname(cls.CONVERSATION_HISTORY_PATH),
#             os.path.dirname(cls.LOG_FILE)
#         ]
        
#         for directory in directories:
#             if directory:
#                 Path(directory).mkdir(parents=True, exist_ok=True)
    
#     @classmethod
#     def validate_config(cls) -> bool:
#         """Valide la configuration complète."""
#         cls.create_directories()
        
#         print("🔍 Validation de la configuration HCP optimisée...")
        
#         # Validation des fichiers de données
#         missing_files = []
#         existing_files = []
        
#         for data_type, file_path in cls.DATA_PATHS.items():
#             if os.path.exists(file_path):
#                 existing_files.append((data_type, file_path))
#                 try:
#                     cls._validate_data_structure(file_path)
#                 except Exception as e:
#                     print(f"⚠️ Structure invalide pour {file_path}: {e}")
#             else:
#                 missing_files.append((data_type, file_path))
        
#         # Fallback vers fichier principal
#         if not existing_files and os.path.exists(cls.DATA_PATH):
#             existing_files.append(('legacy', cls.DATA_PATH))
        
#         # Validation des paramètres
#         validations = [
#             (cls.SIMILARITY_THRESHOLD > 0 and cls.SIMILARITY_THRESHOLD <= 1, "SIMILARITY_THRESHOLD invalide"),
#             (cls.FUZZY_THRESHOLD > 0 and cls.FUZZY_THRESHOLD <= 1, "FUZZY_THRESHOLD invalide"),
#             (cls.BATCH_SIZE > 0, "BATCH_SIZE invalide"),
#             (cls.MAX_LENGTH >= 64, "MAX_LENGTH trop petit"),
#             (len(cls.DEFAULT_RESPONSES) > 0, "DEFAULT_RESPONSES vides"),
#             (len(cls.GREETING_RESPONSES) > 0, "GREETING_RESPONSES vides"),
#             (len(cls.HCP_INDICATOR_DOMAINS) > 0, "HCP_INDICATOR_DOMAINS vides")
#         ]
        
#         validation_passed = True
#         for is_valid, message in validations:
#             if not is_valid:
#                 print(f"❌ {message}")
#                 validation_passed = False
        
#         if validation_passed and existing_files:
#             print("✅ Configuration validée avec succès")
#             print(f"📊 Fichiers de données: {len(existing_files)}")
#             print(f"🎯 Domaines d'indicateurs: {len(cls.HCP_INDICATOR_DOMAINS)}")
#             print(f"🗺️ Aliases territoriaux: {len(cls.TERRITORY_ALIASES)}")
        
#         return validation_passed and len(existing_files) > 0
    
#     @classmethod
#     def _validate_data_structure(cls, file_path: str):
#         """Valide la structure d'un fichier de données."""
#         import json
        
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         if cls.NEW_DATA_STRUCTURE["root_key"] in data:
#             qa_pairs = data[cls.NEW_DATA_STRUCTURE["root_key"]]
#             if not isinstance(qa_pairs, list) or len(qa_pairs) == 0:
#                 raise ValueError("qa_pairs doit être une liste non vide")
            
#             # Validation d'un échantillon
#             sample = qa_pairs[0]
#             required_keys = ["question", "answer", "territoire"]
#             missing_keys = [key for key in required_keys if key not in sample]
#             if missing_keys:
#                 raise ValueError(f"Clés manquantes: {missing_keys}")
    
#     @classmethod
#     def get_search_config_summary(cls) -> Dict[str, Any]:
#         """Résumé de la configuration de recherche."""
#         return {
#             'algorithmes_actifs': [k for k, v in cls.SEARCH_CONFIG.items() if v and k.startswith('use_')],
#             'seuils': {
#                 'semantic': cls.SIMILARITY_THRESHOLD,
#                 'fuzzy': cls.FUZZY_THRESHOLD,
#                 'tfidf': cls.TFIDF_THRESHOLD,
#                 'field_match': cls.FIELD_MATCH_THRESHOLD
#             },
#             'poids_fusion': cls.SEARCH_WEIGHTS,
#             'tfidf_features': cls.TFIDF_CONFIG['max_features'],
#             'territoire_aliases': len(cls.TERRITORY_ALIASES),
#             'domaines_indicateurs': len(cls.HCP_INDICATOR_DOMAINS)
#         }
    
#     @classmethod
#     def get_performance_config_summary(cls) -> Dict[str, Any]:
#         """Résumé de la configuration de performance."""
#         return {
#             'cache': {
#                 'embeddings': cls.ENABLE_EMBEDDING_CACHE,
#                 'recherche': cls.ENABLE_SEARCH_CACHE,
#                 'taille_batch_embedding': cls.EMBEDDING_BATCH_SIZE
#             },
#             'memoire': cls.MEMORY_CONFIG,
#             'systeme': cls.SYSTEM_CONFIG,
#             'metriques': cls.ENABLE_ADVANCED_METRICS
#         }
    
#     @classmethod
#     def is_production_ready(cls) -> bool:
#         """Vérifie si la configuration est prête pour la production."""
#         production_checks = [
#             os.path.exists(cls.DATA_PATH),
#             len(cls.HCP_INDICATOR_DOMAINS) >= 2,
#             len(cls.TERRITORY_ALIASES) > 0,
#             cls.SIMILARITY_THRESHOLD > 0.5,
#             len(cls.DEFAULT_RESPONSES) >= 3,
#             cls.FLASK_PORT > 0
#         ]
        
#         return all(production_checks)

# if __name__ == "__main__":
#     print("=== Validation Configuration HCP Optimisée ===\n")
    
#     # Test de validation
#     is_valid = Config.validate_config()
    
#     if is_valid:
#         print("\n=== Résumé Configuration Recherche ===")
#         search_config = Config.get_search_config_summary()
#         for key, value in search_config.items():
#             print(f"  {key}: {value}")
        
#         print("\n=== Résumé Configuration Performance ===")
#         perf_config = Config.get_performance_config_summary()
#         for key, value in perf_config.items():
#             print(f"  {key}: {value}")
        
#         print(f"\n=== Statut Production ===")
#         if Config.is_production_ready():
#             print("✅ Configuration prête pour la production")
#         else:
#             print("⚠️ Configuration nécessite des ajustements pour la production")
    
#     else:
#         print("❌ Configuration invalide")