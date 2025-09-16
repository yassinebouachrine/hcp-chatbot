# from flask import Flask, render_template, request, jsonify
# from src.chatbot import HCPChatbot, create_chatbot_with_config
# from src.data_processor import HCPDataProcessor
# from config import Config
# import logging
# import os
# import json
# import traceback

# # Configuration du logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)

# # Initialisation de la configuration
# config = Config()

# # Variables globales pour le chatbot
# data_processor = None
# chatbot = None

# def initialize_chatbot():
#     """Initialise le chatbot au démarrage de l'application - adapté à chatbot.py"""
#     global data_processor, chatbot
    
#     try:
#         logger.info("Initialisation du chatbot HCP (structure qa_pairs optimisée)...")
        
#         # Valider la configuration
#         if not config.validate_config():
#             logger.error("Configuration invalide")
#             return False
        
#         # Initialiser le processeur de données
#         logger.info("Initialisation du processeur de données...")
#         data_processor = HCPDataProcessor(config)
        
#         # Charger les données
#         logger.info("Chargement des données...")
#         data = data_processor.load_all_data()
        
#         if data.empty:
#             logger.warning("Aucune donnée chargée, utilisation des FAQ par défaut")
#         else:
#             logger.info(f"Données chargées: {len(data)} entrées")
            
#             # Analyser la structure des données chargées
#             if hasattr(data_processor, 'combined_data') and data_processor.combined_data:
#                 sample_item = data_processor.combined_data[0]
#                 logger.info(f"Champs détectés: {list(sample_item.keys())}")
        
#         # Utiliser la fonction create_chatbot_with_config pour une initialisation cohérente
#         logger.info("Création du chatbot avec configuration...")
#         chatbot = create_chatbot_with_config(config, data_processor)
        
#         if chatbot is None:
#             logger.error("Échec de la création du chatbot")
#             return False
        
#         # Obtenir les statistiques
#         stats = chatbot.get_statistics()
#         logger.info(f"Chatbot initialisé avec succès!")
#         logger.info(f"Statistiques: {stats}")
        
#         # Afficher des informations détaillées
#         logger.info("✅ Chatbot opérationnel avec les caractéristiques suivantes:")
#         logger.info(f"   - Modèle entraîné: {stats.get('is_trained', False)}")
#         logger.info(f"   - Sentence Transformer: {stats.get('has_sentence_transformer', False)}")
#         logger.info(f"   - Paires Q&A: {stats.get('qa_pairs_count', 0)}")
#         logger.info(f"   - Embeddings créés: {stats.get('embedding_count', 0)}")
#         logger.info(f"   - Territoires uniques: {stats.get('unique_territories', 0)}")
#         logger.info(f"   - Indicateurs uniques: {stats.get('unique_indicators', 0)}")
        
#         return True
        
#     except Exception as e:
#         logger.error(f"Erreur lors de l'initialisation du chatbot: {e}")
#         logger.error(f"Trace complète: {traceback.format_exc()}")
#         return False

# @app.route('/')
# def home():
#     """Page d'accueil du chatbot"""
#     try:
#         return render_template('index.html')
#     except Exception as e:
#         logger.error(f"Erreur lors du rendu de la page d'accueil: {e}")
#         return "Erreur lors du chargement de la page", 500

# @app.route('/chat', methods=['POST'])
# def chat():
#     """Endpoint pour les conversations - adapté au chatbot optimisé"""
#     global chatbot
    
#     try:
#         # Vérifier que le chatbot est initialisé
#         if chatbot is None:
#             return jsonify({
#                 'error': 'Chatbot non initialisé',
#                 'status': 'error'
#             }), 500
        
#         data = request.get_json()
        
#         if not data:
#             return jsonify({
#                 'error': 'Données JSON manquantes',
#                 'status': 'error'
#             }), 400
        
#         user_message = data.get('message', '').strip()
        
#         if not user_message:
#             return jsonify({
#                 'error': 'Message vide',
#                 'status': 'error'
#             }), 400
        
#         logger.info(f"Question reçue: {user_message}")
        
#         # Générer la réponse en utilisant la méthode chat du chatbot
#         response = chatbot.chat(user_message)
        
#         logger.info(f"Réponse générée: {response[:100]}...")
        
#         # Métadonnées enrichies basées sur les capacités du chatbot
#         metadata = {
#             'model_used': 'trained' if chatbot.is_trained else 'semantic_search',
#             'qa_pairs_count': len(chatbot.qa_pairs),
#             'response_method': 'semantic_search' if chatbot.sentence_transformer else 'textual_search',
#             'has_embeddings': any('embedding' in pair for pair in chatbot.qa_pairs[:10]),
#             'territory_detected': chatbot.extract_territory_from_query(user_message),
#             'indicator_detected': chatbot.extract_indicator_from_query(user_message),
#             'is_greeting': chatbot.is_greeting(user_message)
#         }
        
#         return jsonify({
#             'response': response,
#             'status': 'success',
#             'metadata': metadata
#         })
        
#     except Exception as e:
#         logger.error(f"Erreur lors du chat: {e}")
#         logger.error(f"Trace: {traceback.format_exc()}")
#         return jsonify({
#             'error': 'Erreur interne du serveur',
#             'status': 'error'
#         }), 500

# @app.route('/health')
# def health():
#     """Endpoint de santé - adapté au chatbot actuel"""
#     global chatbot, data_processor
    
#     try:
#         health_status = {
#             'status': 'healthy',
#             'chatbot_initialized': chatbot is not None,
#             'data_processor_initialized': data_processor is not None,
#             'data_loaded': False,
#             'qa_pairs_count': 0,
#             'embedding_support': False,
#             'model_support': False
#         }
        
#         if data_processor:
#             health_status['data_loaded'] = (
#                 hasattr(data_processor, 'combined_data') and 
#                 data_processor.combined_data is not None and 
#                 len(data_processor.combined_data) > 0
#             )
        
#         if chatbot:
#             health_status.update({
#                 'qa_pairs_count': len(chatbot.qa_pairs),
#                 'embedding_support': chatbot.sentence_transformer is not None,
#                 'model_support': chatbot.model is not None,
#                 'is_trained': chatbot.is_trained
#             })
            
#             # Ajouter les statistiques complètes
#             try:
#                 chatbot_stats = chatbot.get_statistics()
#                 health_status['chatbot_stats'] = chatbot_stats
#             except Exception as e:
#                 logger.warning(f"Impossible d'obtenir les statistiques: {e}")
        
#         return jsonify(health_status)
        
#     except Exception as e:
#         logger.error(f"Erreur lors de la vérification de santé: {e}")
#         return jsonify({
#             'status': 'unhealthy',
#             'error': str(e)
#         }), 500

# @app.route('/stats')
# def stats():
#     """Endpoint pour obtenir les statistiques détaillées"""
#     global chatbot, data_processor
    
#     try:
#         if not chatbot:
#             return jsonify({
#                 'error': 'Chatbot non initialisé',
#                 'status': 'error'
#             }), 500
        
#         # Statistiques du chatbot
#         chatbot_stats = chatbot.get_statistics()
        
#         # Statistiques du processeur de données
#         data_stats = {
#             'has_combined_data': hasattr(data_processor, 'combined_data') and data_processor.combined_data,
#             'data_count': len(data_processor.combined_data) if hasattr(data_processor, 'combined_data') and data_processor.combined_data else 0
#         }
        
#         if hasattr(data_processor, 'get_statistics'):
#             try:
#                 data_processor_stats = data_processor.get_statistics()
#                 data_stats.update(data_processor_stats)
#             except Exception as e:
#                 logger.warning(f"Impossible d'obtenir les statistiques du processeur: {e}")
        
#         # Analyse approfondie des données
#         structure_analysis = {}
#         if hasattr(data_processor, 'combined_data') and data_processor.combined_data:
#             territories = set()
#             indicators = set()
#             genres = set()
            
#             sample_size = min(500, len(data_processor.combined_data))
#             for item in data_processor.combined_data[:sample_size]:
#                 territories.add(item.get('territoire', 'Unknown'))
#                 indicators.add(item.get('indicateur', 'unknown'))
#                 genres.add(item.get('genre', 'ensemble'))
            
#             structure_analysis = {
#                 'territories_count': len(territories),
#                 'indicators_count': len(indicators),
#                 'genres_count': len(genres),
#                 'territories_sample': sorted(list(territories))[:20],
#                 'indicators_sample': sorted(list(indicators))[:20],
#                 'genres_available': sorted(list(genres)),
#                 'sample_size_analyzed': sample_size
#             }
        
#         # Configuration du chatbot
#         configuration = {
#             'base_model': getattr(config, 'BASE_MODEL', 'Unknown'),
#             'embedding_model': getattr(config, 'EMBEDDING_MODEL', 'Unknown'),
#             'similarity_threshold': getattr(config, 'SIMILARITY_THRESHOLD', 'Unknown'),
#             'max_length': getattr(config, 'MAX_LENGTH', 'Unknown'),
#             'temperature': getattr(config, 'TEMPERATURE', 'Unknown'),
#             'model_path': getattr(config, 'MODEL_PATH', 'Unknown'),
#             'data_paths': getattr(config, 'DATA_PATHS', {})
#         }
        
#         stats_data = {
#             'chatbot': chatbot_stats,
#             'data_processor': data_stats,
#             'structure_analysis': structure_analysis,
#             'configuration': configuration,
#             'capabilities': {
#                 'semantic_search': chatbot.sentence_transformer is not None,
#                 'textual_fallback': True,
#                 'filter_search': True,
#                 'greeting_detection': True,
#                 'territory_extraction': True,
#                 'indicator_extraction': True,
#                 'conversation_history': getattr(config, 'SAVE_CONVERSATION_HISTORY', False)
#             }
#         }
        
#         return jsonify(stats_data)
        
#     except Exception as e:
#         logger.error(f"Erreur lors de la récupération des statistiques: {e}")
#         return jsonify({
#             'error': 'Erreur lors de la récupération des statistiques',
#             'status': 'error'
#         }), 500

# @app.route('/help')
# def help():
#     """Endpoint pour obtenir l'aide - utilise le message d'aide du chatbot"""
#     global chatbot
    
#     try:
#         if not chatbot:
#             return jsonify({
#                 'error': 'Chatbot non initialisé',
#                 'status': 'error'
#             }), 500
        
#         # Utiliser le message d'aide intégré du chatbot
#         help_message = chatbot.get_help_message()
        
#         # Exemples dynamiques basés sur les données disponibles
#         examples = [
#             "Quelle est la population légale du Maroc ?",
#             "Population municipale de l'ensemble du territoire national ?",
#             "Quel est le pourcentage de 0-4 ans au niveau national ?",
#             "Pourcentage de la population de 15-19 ans ?",
#             "Qu'est-ce que le HCP ?",
#             "Bonjour, peux-tu m'aider ?"
#         ]
        
#         # Informations sur les capacités disponibles
#         capabilities = {
#             'semantic_search': chatbot.sentence_transformer is not None,
#             'greeting_responses': len(chatbot.greeting_responses) if hasattr(chatbot, 'greeting_responses') else 0,
#             'default_responses': len(chatbot.default_responses) if hasattr(chatbot, 'default_responses') else 0,
#             'qa_pairs_available': len(chatbot.qa_pairs)
#         }
        
#         return jsonify({
#             'help': help_message,
#             'examples': examples,
#             'capabilities': capabilities,
#             'status': 'success'
#         })
        
#     except Exception as e:
#         logger.error(f"Erreur lors de la récupération de l'aide: {e}")
#         return jsonify({
#             'error': 'Erreur lors de la récupération de l\'aide',
#             'status': 'error'
#         }), 500

# @app.route('/territories')
# def territories():
#     """Endpoint pour obtenir la liste des territoires et indicateurs disponibles"""
#     global chatbot, data_processor
    
#     try:
#         territories_list = []
#         indicators_list = []
#         genres_list = []
        
#         # Extraire à partir des données du processeur si disponibles
#         if data_processor and hasattr(data_processor, 'combined_data') and data_processor.combined_data:
#             territories_set = set()
#             indicators_set = set()
#             genres_set = set()
            
#             for item in data_processor.combined_data:
#                 territoire = item.get('territoire', '')
#                 indicateur = item.get('indicateur', '')
#                 genre = item.get('genre', '')
                
#                 if territoire and territoire not in ['Unknown', 'Territoire non spécifié']:
#                     territories_set.add(territoire)
#                 if indicateur and indicateur != 'unknown':
#                     indicators_set.add(indicateur)
#                 if genre and genre != 'non spécifié':
#                     genres_set.add(genre)
            
#             territories_list = sorted(list(territories_set))
#             indicators_list = sorted(list(indicators_set))
#             genres_list = sorted(list(genres_set))
        
#         # Extraire à partir des qa_pairs du chatbot comme fallback
#         elif chatbot and chatbot.qa_pairs:
#             territories_set = set()
#             indicators_set = set()
#             genres_set = set()
            
#             for pair in chatbot.qa_pairs:
#                 territoire = pair.get('territoire', '')
#                 indicateur = pair.get('indicateur', '')
#                 genre = pair.get('genre', '')
                
#                 if territoire and territoire not in ['Unknown', 'Territoire non spécifié']:
#                     territories_set.add(territoire)
#                 if indicateur and indicateur != 'unknown':
#                     indicators_set.add(indicateur)
#                 if genre and genre != 'non spécifié':
#                     genres_set.add(genre)
            
#             territories_list = sorted(list(territories_set))
#             indicators_list = sorted(list(indicators_set))
#             genres_list = sorted(list(genres_set))
        
#         # Ajouter des territoires par défaut s'ils ne sont pas présents
#         default_territories = ['Ensemble du territoire national', 'Maroc']
#         for territory in default_territories:
#             if territory not in territories_list:
#                 territories_list.insert(0, territory)
        
#         return jsonify({
#             'territories': territories_list[:50],  # Limiter à 50
#             'indicators': indicators_list[:30],    # Limiter à 30
#             'genres': genres_list,                 # Tous les genres
#             'counts': {
#                 'territories_count': len(territories_list),
#                 'indicators_count': len(indicators_list),
#                 'genres_count': len(genres_list)
#             },
#             'data_source': 'combined_data' if (data_processor and hasattr(data_processor, 'combined_data')) else 'qa_pairs',
#             'status': 'success'
#         })
        
#     except Exception as e:
#         logger.error(f"Erreur lors de la récupération des territoires: {e}")
#         return jsonify({
#             'error': 'Erreur lors de la récupération des territoires',
#             'status': 'error'
#         }), 500

# @app.route('/search', methods=['POST'])
# def search():
#     """Endpoint pour recherche avancée avec filtres - adapté au chatbot"""
#     global chatbot
    
#     try:
#         if not chatbot:
#             return jsonify({
#                 'error': 'Chatbot non initialisé',
#                 'status': 'error'
#             }), 500
        
#         data = request.get_json()
#         if not data:
#             return jsonify({
#                 'error': 'Données JSON manquantes',
#                 'status': 'error'
#             }), 400
        
#         query = data.get('query', '').strip()
#         filters = data.get('filters', {})
        
#         territoire_filter = filters.get('territoire')
#         indicateur_filter = filters.get('indicateur') 
#         genre_filter = filters.get('genre')
        
#         # Utiliser la méthode search_by_filters du chatbot si disponible
#         if territoire_filter or indicateur_filter:
#             if hasattr(chatbot, 'search_by_filters'):
#                 filtered_result = chatbot.search_by_filters(
#                     territory=territoire_filter,
#                     question_type=None,
#                     indicateur=indicateur_filter
#                 )
                
#                 if filtered_result:
#                     # Vérifier le filtre genre si spécifié
#                     if genre_filter and filtered_result.get('genre') != genre_filter:
#                         filtered_result = None
                
#                 if filtered_result:
#                     return jsonify({
#                         'response': filtered_result['answer'],
#                         'match_info': {
#                             'territoire': filtered_result.get('territoire'),
#                             'indicateur': filtered_result.get('indicateur'),
#                             'genre': filtered_result.get('genre'),
#                             'question': filtered_result.get('question')
#                         },
#                         'search_type': 'filtered_search',
#                         'status': 'success'
#                     })
        
#         # Recherche normale si pas de filtres ou pas de résultat filtré
#         if query:
#             response = chatbot.chat(query)
            
#             # Tentative d'extraction des informations de correspondance
#             match_info = {}
#             if hasattr(chatbot, 'extract_territory_from_query'):
#                 match_info['territory_detected'] = chatbot.extract_territory_from_query(query)
#             if hasattr(chatbot, 'extract_indicator_from_query'):
#                 match_info['indicator_detected'] = chatbot.extract_indicator_from_query(query)
            
#             return jsonify({
#                 'response': response,
#                 'match_info': match_info,
#                 'search_type': 'normal_search',
#                 'status': 'success'
#             })
#         else:
#             return jsonify({
#                 'error': 'Aucune requête ou filtre fourni',
#                 'status': 'error'
#             }), 400
            
#     except Exception as e:
#         logger.error(f"Erreur lors de la recherche: {e}")
#         return jsonify({
#             'error': 'Erreur lors de la recherche',
#             'status': 'error'
#         }), 500

# @app.route('/reset', methods=['POST'])
# def reset_conversation():
#     """Endpoint pour réinitialiser la conversation"""
#     global chatbot
    
#     try:
#         if not chatbot:
#             return jsonify({
#                 'error': 'Chatbot non initialisé',
#                 'status': 'error'
#             }), 500
        
#         # Réinitialiser la conversation (si la méthode existe)
#         if hasattr(chatbot, 'reset_conversation'):
#             chatbot.reset_conversation()
#             message = 'Conversation réinitialisée avec succès'
#         else:
#             message = 'Réinitialisation demandée (pas de méthode reset_conversation)'
        
#         return jsonify({
#             'message': message,
#             'status': 'success'
#         })
        
#     except Exception as e:
#         logger.error(f"Erreur lors de la réinitialisation: {e}")
#         return jsonify({
#             'error': 'Erreur lors de la réinitialisation',
#             'status': 'error'
#         }), 500

# @app.route('/debug', methods=['GET'])
# def debug_info():
#     """Endpoint de débogage complet"""
#     global chatbot, data_processor
    
#     try:
#         debug_data = {
#             'chatbot_initialized': chatbot is not None,
#             'data_processor_initialized': data_processor is not None,
#             'config_info': {
#                 'model_path': getattr(config, 'MODEL_PATH', 'Unknown'),
#                 'data_paths': getattr(config, 'DATA_PATHS', {}),
#                 'embedding_model': getattr(config, 'EMBEDDING_MODEL', 'Unknown')
#             }
#         }
        
#         if data_processor:
#             debug_data['data_processor'] = {
#                 'has_combined_data': hasattr(data_processor, 'combined_data'),
#                 'combined_data_count': len(data_processor.combined_data) if hasattr(data_processor, 'combined_data') and data_processor.combined_data else 0,
#                 'data_sample': data_processor.combined_data[:2] if hasattr(data_processor, 'combined_data') and data_processor.combined_data else []
#             }
        
#         if chatbot:
#             debug_data['chatbot'] = {
#                 'qa_pairs_count': len(chatbot.qa_pairs),
#                 'has_model': chatbot.model is not None,
#                 'has_sentence_transformer': chatbot.sentence_transformer is not None,
#                 'is_trained': chatbot.is_trained,
#                 'qa_pairs_sample': chatbot.qa_pairs[:2] if chatbot.qa_pairs else [],
#                 'has_embeddings': any('embedding' in pair for pair in chatbot.qa_pairs[:5]) if chatbot.qa_pairs else False
#             }
            
#             # Tester les méthodes importantes
#             try:
#                 test_query = "Quelle est la population du Maroc ?"
#                 debug_data['chatbot']['territory_extraction_test'] = chatbot.extract_territory_from_query(test_query)
#                 debug_data['chatbot']['indicator_extraction_test'] = chatbot.extract_indicator_from_query(test_query)
#                 debug_data['chatbot']['greeting_detection_test'] = chatbot.is_greeting("Bonjour")
#             except Exception as e:
#                 debug_data['chatbot']['method_test_error'] = str(e)
        
#         return jsonify(debug_data)
        
#     except Exception as e:
#         logger.error(f"Erreur lors du débogage: {e}")
#         return jsonify({
#             'error': 'Erreur lors du débogage',
#             'debug_error': str(e),
#             'status': 'error'
#         }), 500

# @app.errorhandler(404)
# def not_found(error):
#     """Gestionnaire d'erreur 404"""
#     return jsonify({
#         'error': 'Endpoint non trouvé',
#         'available_endpoints': [
#             'GET /', 'POST /chat', 'GET /health', 'GET /stats', 
#             'GET /help', 'GET /territories', 'POST /search', 
#             'POST /reset', 'GET /debug'
#         ],
#         'status': 'error'
#     }), 404

# @app.errorhandler(500)
# def internal_error(error):
#     """Gestionnaire d'erreur 500"""
#     logger.error(f"Erreur interne du serveur: {error}")
#     return jsonify({
#         'error': 'Erreur interne du serveur',
#         'status': 'error'
#     }), 500

# if __name__ == '__main__':
#     try:
#         # Initialiser le chatbot au démarrage
#         logger.info("Démarrage de l'application HCP Chatbot (version adaptée)...")
        
#         initialization_success = initialize_chatbot()
        
#         if not initialization_success:
#             logger.error("Échec de l'initialisation du chatbot")
#             logger.info("L'application va démarrer mais avec fonctionnalités limitées")
#         else:
#             logger.info("✅ Chatbot initialisé avec succès")
        
#         # Configuration du serveur Flask
#         host = getattr(config, 'FLASK_HOST', '0.0.0.0')
#         port = getattr(config, 'FLASK_PORT', 5000)
#         debug = getattr(config, 'FLASK_DEBUG', True)
        
#         logger.info(f"Démarrage du serveur Flask sur {host}:{port}")
#         logger.info(f"Mode debug: {debug}")
#         logger.info("Endpoints disponibles:")
#         logger.info("  - GET  /           : Page d'accueil")
#         logger.info("  - POST /chat       : Conversation avec le chatbot")
#         logger.info("  - GET  /health     : Statut de santé du système")
#         logger.info("  - GET  /stats      : Statistiques détaillées")
#         logger.info("  - GET  /help       : Aide et guide d'utilisation")
#         logger.info("  - GET  /territories: Territoires/indicateurs disponibles")
#         logger.info("  - POST /search     : Recherche avec filtres")
#         logger.info("  - POST /reset      : Réinitialiser la conversation")
#         logger.info("  - GET  /debug      : Informations de débogage")
        
#         # Configuration de l'application Flask
#         app.config.update({
#             'SECRET_KEY': os.environ.get('SECRET_KEY', 'dev-secret-key'),
#             'JSON_AS_ASCII': False,  # Support UTF-8
#             'JSONIFY_PRETTYPRINT_REGULAR': True
#         })
        
#         app.run(
#             host=host,
#             port=port,
#             debug=debug,
#             use_reloader=False  # Éviter les problèmes de double initialisation
#         )
        
#     except KeyboardInterrupt:
#         logger.info("Arrêt de l'application par l'utilisateur")
#     except Exception as e:
#         logger.error(f"Erreur fatale lors du démarrage: {e}")
#         logger.error(f"Trace: {traceback.format_exc()}")
#         raise









# # -----------------------------
# # app adapted module (Flask)
# # -----------------------------
# from flask import Flask, render_template, request, jsonify
# import traceback

# app = Flask(__name__)

# # Globals
# _data_processor = None
# _chatbot = None
# _config = None


# def initialize_chatbot_app(config):
#     """Initialise DataProcessor et HCPChatbotAdapted pour l'app Flask"""
#     global _data_processor, _chatbot, _config
#     _config = config
#     try:
#         from src.data_processor import HCPDataProcessor
#         from src.chatbot import HCPChatbotAdapted
#     except Exception:
#         # fallback if module path different
#         from src.data_processor import HCPDataProcessor

#     _data_processor = HCPDataProcessor(config)
#     df = _data_processor.load_all_data()
#     # create chatbot
#     _chatbot = HCPChatbotAdapted(config, _data_processor)
#     _chatbot.load_model()
#     _chatbot.initialize_qa_pairs()
#     return True


# @app.route('/')
# def home():
#     try:
#         return render_template('index.html')
#     except Exception as e:
#         return "Index render error", 500


# @app.route('/chat', methods=['POST'])
# def chat_route():
#     global _chatbot
#     try:
#         if _chatbot is None:
#             return jsonify({'error': 'Chatbot not initialized', 'status': 'error'}), 500
#         data = request.get_json()
#         if not data:
#             return jsonify({'error': 'No JSON provided', 'status': 'error'}), 400
#         user_message = data.get('message','').strip()
#         if not user_message:
#             return jsonify({'error': 'Empty message', 'status': 'error'}), 400
#         response = _chatbot.chat(user_message)
#         metadata = {
#             'model_used': 'trained' if _chatbot.is_trained else 'semantic_search',
#             'qa_pairs_count': len(_chatbot.qa_pairs),
#             'response_method': 'semantic_search' if _chatbot.sentence_transformer else 'textual_search',
#             'territory_detected': _chatbot.extract_territory_from_query(user_message),
#             'indicator_detected': _chatbot.extract_indicator_from_query(user_message),
#             'is_greeting': _chatbot.is_greeting(user_message)
#         }
#         return jsonify({'response': response, 'status': 'success', 'metadata': metadata})
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# @app.route('/health')
# def health():
#     global _chatbot, _data_processor
#     try:
#         status = {'status': 'healthy', 'chatbot_initialized': _chatbot is not None}
#         if _data_processor and hasattr(_data_processor, 'combined_data'):
#             status.update({'data_loaded': len(_data_processor.combined_data) > 0, 'data_count': len(_data_processor.combined_data)})
#         if _chatbot:
#             status.update({'qa_pairs_count': len(_chatbot.qa_pairs), 'is_trained': _chatbot.is_trained,
#                            'has_embeddings': any('embedding' in p for p in _chatbot.qa_pairs[:20])})
#         return jsonify(status)
#     except Exception as e:
#         return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


# @app.route('/territories')
# def territories_route():
#     global _data_processor, _chatbot
#     try:
#         territories_set = set()
#         indicators_set = set()
#         genres_set = set()
#         if _data_processor and hasattr(_data_processor, 'combined_data') and _data_processor.combined_data:
#             for item in _data_processor.combined_data:
#                 territories_set.add(item.get('territory') or item.get('original_territory') or item.get('territoire') or '')
#                 indicators_set.add(item.get('variable') or item.get('indicateur') or '')
#                 genres_set.add(item.get('sexe') or item.get('genre') or '')
#         elif _chatbot and _chatbot.qa_pairs:
#             for p in _chatbot.qa_pairs:
#                 territories_set.add(p.get('territory') or p.get('territoire') or '')
#                 indicators_set.add(p.get('variable') or p.get('indicateur') or '')
#                 genres_set.add(p.get('sexe') or p.get('genre') or '')

#         territories = sorted([t for t in territories_set if t])
#         indicators = sorted([i for i in indicators_set if i])
#         genres = sorted([g for g in genres_set if g])
#         # ensure default
#         for d in ['Ensemble du territoire national','Maroc']:
#             if d not in territories:
#                 territories.insert(0,d)
#         return jsonify({'territories': territories[:50], 'indicators': indicators[:30], 'genres': genres, 'status': 'success'})
#     except Exception as e:
#         return jsonify({'error': str(e), 'status': 'error'}), 500


# # expose initializer for the user's app to call
# def create_app_with_config(config):
#     initialize_chatbot_app(config)
#     return app


# if __name__ == '__main__':
#     # if executed directly, try to import Config
#     try:
#         from config import Config
#         conf = Config()
#         create_app_with_config(conf)
#         host = getattr(conf, 'FLASK_HOST', '0.0.0.0')
#         port = getattr(conf, 'FLASK_PORT', 5000)
#         debug = getattr(conf, 'FLASK_DEBUG', True)
#         app.run(host=host, port=port, debug=debug, use_reloader=False)
#     except Exception as e:
#         print('Erreur démarrage app:', e)








from flask import Flask, render_template, request, jsonify
import traceback
import logging
import time
from typing import Optional, Dict

app = Flask(__name__)

# Globals
_data_processor = None
_chatbot = None
_config = None
_app_stats = {
    'start_time': time.time(),
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'avg_response_time': 0.0,
    'response_times': []
}

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('FlaskAppOptimized')


def _safe_call(obj, method_name, *args, default=None):
    """Appelle une méthode si elle existe, sinon renvoie default."""
    try:
        func = getattr(obj, method_name, None)
        if callable(func):
            return func(*args)
    except Exception as e:
        logger.debug(f"Erreur appel sécurisé {method_name}: {e}")
    return default


def initialize_chatbot_app(config):
    """Initialise DataProcessor et HCPChatbotOptimized pour l'app Flask."""
    global _data_processor, _chatbot, _config
    _config = config

    try:
        logger.info("Initialisation du chatbot optimisé...")

        # Import des modules optimisés (chemins flexibles)
        HCPDataProcessor = None
        HCPChatbotOptimized = None
        try:
            from src.data_processor import HCPDataProcessor
        except Exception:
            try:
                from data_processor import HCPDataProcessor
            except Exception:
                logger.warning("HCPDataProcessor non trouvé dans src/ ni racine. Continuer sans lui.")

        try:
            # tenter différents chemins pour la classe du chatbot (nom au cas où)
            try:
                from src.chatbot import HCPChatbotOptimized
            except Exception:
                from chatbot import HCPChatbotOptimized
        except Exception as e:
            logger.error(f"Impossible d'importer HCPChatbotOptimized: {e}")
            traceback.print_exc()
            return False

        # Initialisation du data processor si la classe est disponible
        if HCPDataProcessor:
            logger.info("Initialisation du processeur de données...")
            try:
                _data_processor = HCPDataProcessor(config)
                # load_all_data peut retourner un dataframe ou une liste
                df = _safe_call(_data_processor, 'load_all_data', default=None)
                loaded_count = 0
                if df is None:
                    logger.info("Aucune donnée retournée par load_all_data()")
                else:
                    try:
                        loaded_count = len(df)
                    except Exception:
                        loaded_count = 1
                logger.info(f"Données chargées: {loaded_count} entrées")
            except Exception as e:
                logger.warning(f"Erreur initialisation data_processor: {e}")
                _data_processor = None
        else:
            logger.info("Pas de HCPDataProcessor installé; l'initialisation continuera sans données.")

        # Création du chatbot optimisé
        logger.info("Création du chatbot optimisé...")
        try:
            _chatbot = HCPChatbotOptimized(config, _data_processor)
        except Exception as e:
            logger.error(f"Erreur lors de l'instanciation du chatbot: {e}")
            traceback.print_exc()
            return False

        # Chargement du modèle (si présent)
        try:
            _safe_call(_chatbot, 'load_model')
            logger.info(f"Modèle chargé: {'Entraîné' if getattr(_chatbot, 'is_trained', False) else 'Mode recherche sémantique'}")
        except Exception as e:
            logger.warning(f"Erreur chargement modèle: {e}")

        # Initialisation des paires Q&A
        try:
            _safe_call(_chatbot, 'initialize_qa_pairs')
            stats = _safe_call(_chatbot, 'get_statistics', default={}) or {}
            logger.info("Chatbot initialisé avec succès:")
            logger.info(f"  - {stats.get('qa_pairs_count', len(getattr(_chatbot, 'qa_pairs', [])))} paires Q&A")
            logger.info(f"  - {stats.get('embedding_count', sum(1 for p in getattr(_chatbot, 'qa_pairs', []) if 'embedding' in p))} embeddings")
            logger.info(f"  - {stats.get('vocabulary_size', len(getattr(_chatbot, 'vocabulary', [])))} mots dans le vocabulaire")
            logger.info(f"  - Index de recherche: {stats.get('search_index_size', len(getattr(_chatbot, 'search_index', {})))} entrées")
        except Exception as e:
            logger.error(f"Erreur initialisation paires Q&A: {e}")

        return True

    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation: {e}")
        traceback.print_exc()
        return False


def _update_request_stats(response_time: float, success: bool):
    """Met à jour les statistiques de requête."""
    global _app_stats

    _app_stats['total_requests'] += 1
    if success:
        _app_stats['successful_requests'] += 1
    else:
        _app_stats['failed_requests'] += 1

    _app_stats['response_times'].append(response_time)
    if len(_app_stats['response_times']) > 1000:
        _app_stats['response_times'] = _app_stats['response_times'][-1000:]

    if _app_stats['response_times']:
        _app_stats['avg_response_time'] = sum(_app_stats['response_times']) / len(_app_stats['response_times'])


@app.route('/')
def home():
    """Page d'accueil avec gestion d'erreur améliorée."""
    try:
        logger.debug("Affichage de la page d'accueil")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Erreur rendu template index.html: {e}")
        return jsonify({
            'error': 'Erreur lors du chargement de la page d\'accueil',
            'status': 'error'
        }), 500


@app.route('/chat', methods=['POST'])
def chat_route():
    """Route de chat optimisée avec métadonnées enrichies."""
    global _chatbot
    start_time = time.time()
    success = False

    try:
        if _chatbot is None:
            logger.error("Tentative d'utilisation du chatbot non initialisé")
            return jsonify({
                'error': 'Chatbot non initialisé',
                'status': 'error',
                'suggestion': 'Veuillez redémarrer l\'application'
            }), 500

        data = request.get_json()
        if not data:
            logger.warning("Requête sans données JSON")
            return jsonify({
                'error': 'Aucune donnée JSON fournie',
                'status': 'error'
            }), 400

        user_message = (data.get('message') or '').strip()
        if not user_message:
            logger.warning("Message vide reçu")
            return jsonify({
                'error': 'Message vide',
                'status': 'error'
            }), 400

        logger.info(f"Question reçue: {user_message[:100]}{'...' if len(user_message) > 100 else ''}")

        processing_start = time.time()
        response = _safe_call(_chatbot, 'chat', user_message, default=None)
        processing_time = time.time() - processing_start

        if response is None:
            logger.error("Le chatbot n'a pas retourné de réponse valide")
            return jsonify({'error': 'Aucune réponse', 'status': 'error'}), 500

        # Collecte de métadonnées protégées
        metadata = {
            'model_used': 'trained' if getattr(_chatbot, 'is_trained', False) else 'search_only',
            'response_method': _get_response_method(_chatbot),
            'processing_time_ms': round(processing_time * 1000, 2),
            'qa_pairs_count': len(getattr(_chatbot, 'qa_pairs', [])),
            'has_embeddings': getattr(_chatbot, 'sentence_transformer', None) is not None,
            'vocabulary_size': len(getattr(_chatbot, 'vocabulary', [])),
            'territory_detected': _safe_call(_chatbot, 'extract_territory_from_query', user_message, default=None),
            'indicator_detected': _safe_call(_chatbot, 'extract_indicator_from_query', user_message, default=None),
            'source_detected': _safe_call(_chatbot, 'extract_source_from_query', user_message, default=None),
            'is_greeting': _safe_call(_chatbot, 'is_greeting', user_message, default=False),
            'query_corrected': _safe_call(_chatbot, '_spell_correct_query', user_message, default=None) if hasattr(_chatbot, '_spell_correct_query') else None,
            'response_length': len(response),
            'confidence_score': _estimate_confidence(user_message, response, _chatbot)
        }

        logger.info(f"Réponse générée en {processing_time:.3f}s, méthode: {metadata['response_method']}")
        if metadata.get('territory_detected'):
            logger.debug(f"Territoire détecté: {metadata['territory_detected']}")
        if metadata.get('indicator_detected'):
            logger.debug(f"Indicateur détecté: {metadata['indicator_detected']}")

        success = True
        return jsonify({
            'response': response,
            'status': 'success',
            'metadata': metadata
        })

    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête: {e}")
        traceback.print_exc()
        return jsonify({
            'error': 'Erreur interne du serveur',
            'status': 'error',
            'details': str(e) if _config and getattr(_config, 'DEBUG', False) else None
        }), 500

    finally:
        total_time = time.time() - start_time
        _update_request_stats(total_time, success)


def _get_response_method(chatbot) -> str:
    """Détermine la méthode de réponse utilisée."""
    if not chatbot:
        return 'unknown'
    if getattr(chatbot, 'is_trained', False) and getattr(chatbot, 'model', None):
        return 'trained_model'
    if getattr(chatbot, 'sentence_transformer', None):
        return 'semantic_search'
    if getattr(chatbot, 'search_index', None):
        return 'index_search'
    return 'text_matching'


def _estimate_confidence(query: str, response: str, chatbot) -> float:
    """Estime un score de confiance pour la réponse."""
    confidence = 0.5
    try:
        if _safe_call(chatbot, 'extract_territory_from_query', query):
            confidence += 0.2
        if _safe_call(chatbot, 'extract_indicator_from_query', query):
            confidence += 0.2
        if _safe_call(chatbot, 'extract_source_from_query', query):
            confidence += 0.1

        default_responses = getattr(chatbot, 'default_responses', []) or []
        if any(d in response for d in default_responses):
            confidence -= 0.3

        if 50 <= len(response) <= 300:
            confidence += 0.1
        if getattr(chatbot, 'is_trained', False):
            confidence += 0.1

        return max(0.0, min(1.0, confidence))
    except Exception:
        return 0.5


@app.route('/health')
def health():
    """Endpoint de santé avec diagnostic détaillé."""
    global _chatbot, _data_processor, _app_stats

    try:
        status = {
            'status': 'healthy',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'uptime_seconds': round(time.time() - _app_stats['start_time'], 2),
            'chatbot_initialized': _chatbot is not None
        }

        status['app_stats'] = {
            'total_requests': _app_stats['total_requests'],
            'successful_requests': _app_stats['successful_requests'],
            'failed_requests': _app_stats['failed_requests'],
            'success_rate': (_app_stats['successful_requests'] / max(_app_stats['total_requests'], 1)) * 100,
            'avg_response_time_ms': round(_app_stats['avg_response_time'] * 1000, 2)
        }

        if _data_processor and hasattr(_data_processor, 'combined_data'):
            status['data'] = {
                'loaded': len(_data_processor.combined_data) > 0,
                'count': len(_data_processor.combined_data),
                'source': 'HCPDataProcessor'
            }

        if _chatbot:
            chatbot_stats = _safe_call(_chatbot, 'get_statistics', default={}) or {}
            perf_report = _safe_call(_chatbot, 'get_performance_report', default={}) or {}
            status['chatbot'] = {
                'qa_pairs_count': chatbot_stats.get('qa_pairs_count', len(getattr(_chatbot, 'qa_pairs', []))),
                'is_trained': chatbot_stats.get('is_trained', getattr(_chatbot, 'is_trained', False)),
                'has_sentence_transformer': chatbot_stats.get('has_sentence_transformer', getattr(_chatbot, 'sentence_transformer', None) is not None),
                'embedding_count': chatbot_stats.get('embedding_count', sum(1 for p in getattr(_chatbot, 'qa_pairs', []) if 'embedding' in p)),
                'vocabulary_size': chatbot_stats.get('vocabulary_size', len(getattr(_chatbot, 'vocabulary', []))),
                'search_index_size': chatbot_stats.get('search_index_size', len(getattr(_chatbot, 'search_index', {}))),
                'unique_territories': chatbot_stats.get('unique_territories', 0),
                'unique_indicators': chatbot_stats.get('unique_indicators', 0),
                'unique_sources': chatbot_stats.get('unique_sources', 0),
                'data_structure': chatbot_stats.get('data_structure', 'unknown'),
                'performance_report': perf_report
            }

            try:
                test_response = _safe_call(_chatbot, 'chat', 'Bonjour', default='')
                status['chatbot']['test_passed'] = bool(test_response)
            except Exception as e:
                status['chatbot']['test_passed'] = False
                status['chatbot']['test_error'] = str(e)

        critical_issues = []
        if not _chatbot:
            critical_issues.append("Chatbot non initialisé")
        elif not getattr(_chatbot, 'qa_pairs', []):
            critical_issues.append("Aucune paire Q&A chargée")
        if not _data_processor:
            critical_issues.append("Processeur de données non initialisé")
        if critical_issues:
            status['status'] = 'unhealthy'
            status['critical_issues'] = critical_issues

        return jsonify(status)

    except Exception as e:
        logger.error(f"Erreur dans l'endpoint health: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }), 500


@app.route('/territories')
def territories_route():
    """Endpoint pour récupérer les territoires, indicateurs et genres disponibles."""
    global _data_processor, _chatbot

    try:
        territories_set = set()
        indicators_set = set()
        genres_set = set()
        sources_set = set()

        if _data_processor and hasattr(_data_processor, 'combined_data') and _data_processor.combined_data:
            for item in _data_processor.combined_data:
                territory = (item.get('territory') or item.get('original_territory') or item.get('territoire') or '').strip()
                if territory:
                    territories_set.add(territory)
                indicator = (item.get('variable') or item.get('indicateur') or item.get('column_original') or '').strip()
                if indicator:
                    indicators_set.add(indicator)
                genre = (item.get('sexe') or item.get('genre') or '').strip()
                if genre:
                    genres_set.add(genre)
                source = (item.get('source_data') or item.get('source') or '').strip()
                if source:
                    sources_set.add(source)

        elif _chatbot and getattr(_chatbot, 'qa_pairs', None):
            for pair in _chatbot.qa_pairs:
                territory = (pair.get('territory') or pair.get('territoire') or '').strip()
                if territory:
                    territories_set.add(territory)
                indicator = (pair.get('variable') or pair.get('indicateur') or '').strip()
                if indicator:
                    indicators_set.add(indicator)
                genre = (pair.get('sexe') or pair.get('genre') or '').strip()
                if genre:
                    genres_set.add(genre)
                source = (pair.get('source_data') or pair.get('source') or '').strip()
                if source:
                    sources_set.add(source)

        territories = sorted([t for t in territories_set if t and t != 'Unknown'])
        indicators = sorted([i for i in indicators_set if i and i != 'unknown'])
        genres = sorted([g for g in genres_set if g])
        sources = sorted([s for s in sources_set if s and s != 'non spécifié'])

        default_territories = ['Ensemble du territoire national', 'Maroc']
        for default_territory in default_territories:
            if default_territory not in territories:
                territories.insert(0, default_territory)

        if 'ensemble' in genres:
            genres.remove('ensemble')
            genres.insert(0, 'ensemble')

        result = {
            'territories': territories[:50],
            'indicators': indicators[:30],
            'genres': genres,
            'sources': sources[:20],
            'status': 'success',
            'counts': {
                'total_territories': len(territories_set),
                'total_indicators': len(indicators_set),
                'total_genres': len(genres_set),
                'total_sources': len(sources_set)
            }
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Erreur dans territories_route: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/performance')
def performance_route():
    """Endpoint pour les métriques de performance détaillées."""
    global _chatbot, _app_stats

    try:
        if not _chatbot:
            return jsonify({'error': 'Chatbot non initialisé', 'status': 'error'}), 500

        performance_report = _safe_call(_chatbot, 'get_performance_report', default={}) or {}

        app_performance = {
            'uptime_hours': round((time.time() - _app_stats['start_time']) / 3600, 2),
            'total_requests': _app_stats['total_requests'],
            'success_rate': round((_app_stats['successful_requests'] / max(_app_stats['total_requests'], 1)) * 100, 2),
            'avg_response_time_ms': round(_app_stats['avg_response_time'] * 1000, 2),
            'requests_per_hour': round(_app_stats['total_requests'] / max((time.time() - _app_stats['start_time']) / 3600, 0.01), 2)
        }

        chatbot_stats = _safe_call(_chatbot, 'get_statistics', default={}) or {}

        return jsonify({
            'status': 'success',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'app_performance': app_performance,
            'chatbot_performance': performance_report,
            'chatbot_stats': chatbot_stats
        })

    except Exception as e:
        logger.error(f"Erreur dans performance_route: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/debug')
def debug_route():
    """Endpoint de debug pour le développement (à désactiver en production)."""
    if not _config or not getattr(_config, 'DEBUG', False):
        return jsonify({'error': 'Debug non activé', 'status': 'error'}), 403

    try:
        debug_info = {
            'config': {
                'embedding_model': getattr(_config, 'EMBEDDING_MODEL', 'Non défini'),
                'model_path': getattr(_config, 'MODEL_PATH', 'Non défini'),
                'similarity_threshold': getattr(_config, 'SIMILARITY_THRESHOLD', 'Non défini'),
                'flask_debug': getattr(_config, 'FLASK_DEBUG', False)
            },
            'chatbot_state': {
                'initialized': _chatbot is not None,
                'model_loaded': getattr(_chatbot, 'model', None) is not None if _chatbot else False,
                'sentence_transformer_loaded': getattr(_chatbot, 'sentence_transformer', None) is not None if _chatbot else False,
                'qa_pairs_count': len(getattr(_chatbot, 'qa_pairs', [])) if _chatbot else 0
            },
            'memory_info': {
                'vocabulary_size': len(getattr(_chatbot, 'vocabulary', [])) if _chatbot else 0,
                'search_index_size': len(getattr(_chatbot, 'search_index', {})) if _chatbot else 0,
                'embeddings_cache_size': len(getattr(_chatbot, 'embeddings_cache', {})) if _chatbot else 0
            }
        }

        return jsonify(debug_info)

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


# Gestionnaire d'erreur global
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint non trouvé', 'status': 'error'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Erreur 500: {error}")
    return jsonify({'error': 'Erreur interne du serveur', 'status': 'error'}), 500


# Fonction principale pour créer l'app avec configuration
def create_app_with_config(config):
    global _config
    _config = config
    if hasattr(config, 'SECRET_KEY'):
        app.secret_key = config.SECRET_KEY
    success = initialize_chatbot_app(config)
    if not success:
        logger.error("Échec de l'initialisation du chatbot")
    else:
        logger.info("Application Flask initialisée avec succès")
    return app


# Point d'entrée principal
if __name__ == '__main__':
    try:
        from config import Config
        conf = Config()
        create_app_with_config(conf)
        host = getattr(conf, 'FLASK_HOST', '0.0.0.0')
        port = getattr(conf, 'FLASK_PORT', 5000)
        debug = getattr(conf, 'FLASK_DEBUG', True)
        logger.info(f"Démarrage du serveur Flask sur {host}:{port} (debug={debug})")
        app.run(host=host, port=port, debug=debug, use_reloader=False)
    except Exception as e:
        logger.error(f"Erreur lors du démarrage de l'application: {e}")
        traceback.print_exc()












# from flask import Flask, render_template, request, jsonify, g
# import traceback
# import logging
# import time
# import threading
# from typing import Optional, Dict, List, Any
# from dataclasses import dataclass, asdict
# from functools import wraps
# import json
# import os
# from collections import defaultdict, deque
# from datetime import datetime, timedelta

# app = Flask(__name__)

# # Variables globales
# _data_processor = None
# _chatbot = None
# _config = None

# @dataclass
# class RequestMetrics:
#     """Structure pour les métriques de requête."""
#     timestamp: float
#     endpoint: str
#     method: str
#     response_time: float
#     status_code: int
#     user_agent: str = ""
#     ip_address: str = ""
#     query_length: int = 0
#     response_length: int = 0
#     territory_detected: bool = False
#     indicator_detected: bool = False
#     search_method: str = ""
#     confidence_score: float = 0.0

# class AdvancedMetricsCollector:
#     """Collecteur de métriques avancé avec analyse en temps réel."""
    
#     def __init__(self, max_history: int = 10000):
#         self.max_history = max_history
#         self.requests_history = deque(maxlen=max_history)
#         self.real_time_stats = {
#             'total_requests': 0,
#             'successful_requests': 0,
#             'failed_requests': 0,
#             'avg_response_time': 0.0,
#             'requests_per_minute': 0.0,
#             'error_rate': 0.0
#         }
#         self.endpoint_stats = defaultdict(lambda: {
#             'count': 0,
#             'avg_time': 0.0,
#             'error_count': 0,
#             'last_accessed': None
#         })
#         self.error_patterns = defaultdict(int)
#         self.performance_alerts = []
#         self._lock = threading.RLock()
        
#     def add_request(self, metrics: RequestMetrics):
#         """Ajoute une nouvelle métrique de requête."""
#         with self._lock:
#             self.requests_history.append(metrics)
#             self._update_real_time_stats()
#             self._update_endpoint_stats(metrics)
#             self._check_performance_alerts(metrics)
    
#     def _update_real_time_stats(self):
#         """Met à jour les statistiques en temps réel."""
#         if not self.requests_history:
#             return
        
#         recent_requests = [r for r in self.requests_history 
#                           if time.time() - r.timestamp < 300]  # 5 minutes
        
#         total = len(recent_requests)
#         successful = len([r for r in recent_requests if r.status_code < 400])
#         failed = total - successful
        
#         if total > 0:
#             avg_time = sum(r.response_time for r in recent_requests) / total
#             error_rate = (failed / total) * 100
#         else:
#             avg_time = 0.0
#             error_rate = 0.0
        
#         # Calcul des requêtes par minute
#         one_minute_ago = time.time() - 60
#         recent_minute = [r for r in recent_requests if r.timestamp > one_minute_ago]
        
#         self.real_time_stats.update({
#             'total_requests': len(self.requests_history),
#             'successful_requests': len([r for r in self.requests_history if r.status_code < 400]),
#             'failed_requests': len([r for r in self.requests_history if r.status_code >= 400]),
#             'avg_response_time': avg_time,
#             'requests_per_minute': len(recent_minute),
#             'error_rate': error_rate
#         })
    
#     def _update_endpoint_stats(self, metrics: RequestMetrics):
#         """Met à jour les statistiques par endpoint."""
#         endpoint = metrics.endpoint
#         stats = self.endpoint_stats[endpoint]
        
#         stats['count'] += 1
#         stats['last_accessed'] = datetime.fromtimestamp(metrics.timestamp)
        
#         # Calcul de la moyenne mobile
#         current_avg = stats['avg_time']
#         new_avg = ((current_avg * (stats['count'] - 1)) + metrics.response_time) / stats['count']
#         stats['avg_time'] = new_avg
        
#         if metrics.status_code >= 400:
#             stats['error_count'] += 1
    
#     def _check_performance_alerts(self, metrics: RequestMetrics):
#         """Vérifie et génère des alertes de performance."""
#         alerts = []
        
#         # Alerte temps de réponse élevé
#         if metrics.response_time > 5.0:
#             alerts.append({
#                 'type': 'HIGH_RESPONSE_TIME',
#                 'message': f'Temps de réponse élevé: {metrics.response_time:.2f}s',
#                 'endpoint': metrics.endpoint,
#                 'timestamp': metrics.timestamp
#             })
        
#         # Alerte taux d'erreur élevé
#         if self.real_time_stats['error_rate'] > 20.0:
#             alerts.append({
#                 'type': 'HIGH_ERROR_RATE',
#                 'message': f'Taux d\'erreur élevé: {self.real_time_stats["error_rate"]:.1f}%',
#                 'timestamp': metrics.timestamp
#             })
        
#         # Alerte charge élevée
#         if self.real_time_stats['requests_per_minute'] > 100:
#             alerts.append({
#                 'type': 'HIGH_LOAD',
#                 'message': f'Charge élevée: {self.real_time_stats["requests_per_minute"]} req/min',
#                 'timestamp': metrics.timestamp
#             })
        
#         self.performance_alerts.extend(alerts)
#         # Garder seulement les 100 dernières alertes
#         self.performance_alerts = self.performance_alerts[-100:]
    
#     def get_comprehensive_report(self) -> Dict[str, Any]:
#         """Génère un rapport complet des métriques."""
#         with self._lock:
#             # Analyse temporelle
#             now = time.time()
#             time_ranges = {
#                 'last_hour': now - 3600,
#                 'last_day': now - 86400,
#                 'last_week': now - 604800
#             }
            
#             temporal_analysis = {}
#             for period, threshold in time_ranges.items():
#                 period_requests = [r for r in self.requests_history if r.timestamp > threshold]
#                 if period_requests:
#                     temporal_analysis[period] = {
#                         'request_count': len(period_requests),
#                         'avg_response_time': sum(r.response_time for r in period_requests) / len(period_requests),
#                         'error_count': len([r for r in period_requests if r.status_code >= 400]),
#                         'unique_ips': len(set(r.ip_address for r in period_requests if r.ip_address))
#                     }
#                 else:
#                     temporal_analysis[period] = {
#                         'request_count': 0,
#                         'avg_response_time': 0.0,
#                         'error_count': 0,
#                         'unique_ips': 0
#                     }
            
#             # Analyse des patterns d'utilisation
#             usage_patterns = self._analyze_usage_patterns()
            
#             return {
#                 'real_time_stats': self.real_time_stats.copy(),
#                 'endpoint_stats': dict(self.endpoint_stats),
#                 'temporal_analysis': temporal_analysis,
#                 'usage_patterns': usage_patterns,
#                 'recent_alerts': self.performance_alerts[-10:],
#                 'system_health': self._calculate_system_health()
#             }
    
#     def _analyze_usage_patterns(self) -> Dict[str, Any]:
#         """Analyse les patterns d'utilisation."""
#         if not self.requests_history:
#             return {}
        
#         # Analyse des territoires les plus demandés
#         territory_requests = [r for r in self.requests_history if r.territory_detected]
#         territory_count = len(territory_requests)
        
#         # Analyse des méthodes de recherche
#         search_methods = defaultdict(int)
#         confidence_scores = []
        
#         for request in self.requests_history:
#             if request.search_method:
#                 search_methods[request.search_method] += 1
#             if request.confidence_score > 0:
#                 confidence_scores.append(request.confidence_score)
        
#         return {
#             'territory_detection_rate': (territory_count / len(self.requests_history)) * 100,
#             'avg_confidence_score': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
#             'search_method_distribution': dict(search_methods),
#             'avg_query_length': sum(r.query_length for r in self.requests_history) / len(self.requests_history),
#             'response_size_distribution': {
#                 'avg': sum(r.response_length for r in self.requests_history) / len(self.requests_history),
#                 'min': min(r.response_length for r in self.requests_history) if self.requests_history else 0,
#                 'max': max(r.response_length for r in self.requests_history) if self.requests_history else 0
#             }
#         }
    
#     def _calculate_system_health(self) -> Dict[str, Any]:
#         """Calcule l'état de santé du système."""
#         if not self.requests_history:
#             return {'status': 'unknown', 'score': 0}
        
#         recent_requests = [r for r in self.requests_history 
#                           if time.time() - r.timestamp < 300]  # 5 minutes
        
#         if not recent_requests:
#             return {'status': 'idle', 'score': 100}
        
#         # Calcul du score de santé (0-100)
#         health_factors = []
        
#         # Facteur temps de réponse
#         avg_response_time = sum(r.response_time for r in recent_requests) / len(recent_requests)
#         response_score = max(0, 100 - (avg_response_time * 20))  # Pénalité si > 5s
#         health_factors.append(('response_time', response_score))
        
#         # Facteur taux d'erreur
#         error_count = len([r for r in recent_requests if r.status_code >= 400])
#         error_rate = (error_count / len(recent_requests)) * 100
#         error_score = max(0, 100 - (error_rate * 2))  # Pénalité forte pour erreurs
#         health_factors.append(('error_rate', error_score))
        
#         # Facteur charge système
#         load = len(recent_requests) / 5  # Requêtes par minute
#         load_score = 100 if load < 10 else max(0, 100 - ((load - 10) * 5))
#         health_factors.append(('load', load_score))
        
#         overall_score = sum(score for _, score in health_factors) / len(health_factors)
        
#         if overall_score >= 90:
#             status = 'excellent'
#         elif overall_score >= 75:
#             status = 'good'
#         elif overall_score >= 50:
#             status = 'warning'
#         else:
#             status = 'critical'
        
#         return {
#             'status': status,
#             'score': round(overall_score, 1),
#             'factors': dict(health_factors),
#             'recommendations': self._generate_health_recommendations(health_factors)
#         }
    
#     def _generate_health_recommendations(self, health_factors: List) -> List[str]:
#         """Génère des recommandations d'amélioration."""
#         recommendations = []
        
#         for factor_name, score in health_factors:
#             if score < 70:
#                 if factor_name == 'response_time':
#                     recommendations.append("Optimiser les temps de réponse - considérer la mise en cache")
#                 elif factor_name == 'error_rate':
#                     recommendations.append("Réduire le taux d'erreur - vérifier les logs d'erreurs")
#                 elif factor_name == 'load':
#                     recommendations.append("Gérer la charge système - envisager la mise à l'échelle")
        
#         return recommendations


# # Instance globale du collecteur de métriques
# metrics_collector = AdvancedMetricsCollector()

# def track_request_metrics(f):
#     """Décorateur pour traquer les métriques de requête."""
#     @wraps(f)
#     def decorated_function(*args, **kwargs):
#         start_time = time.time()
#         endpoint = request.endpoint or 'unknown'
        
#         try:
#             response = f(*args, **kwargs)
#             status_code = 200
            
#             # Extraction des métriques spécifiques au chat
#             territory_detected = False
#             indicator_detected = False
#             search_method = ""
#             confidence_score = 0.0
#             query_length = 0
#             response_length = 0
            
#             if hasattr(response, 'get_json') and callable(response.get_json):
#                 try:
#                     json_data = response.get_json()
#                     if json_data and isinstance(json_data, dict):
#                         metadata = json_data.get('metadata', {})
#                         territory_detected = bool(metadata.get('territory_detected'))
#                         indicator_detected = bool(metadata.get('indicator_detected'))
#                         search_method = metadata.get('response_method', '')
#                         confidence_score = metadata.get('confidence_score', 0.0)
#                         response_length = metadata.get('response_length', 0)
#                 except:
#                     pass
            
#             # Longueur de la requête
#             if request.is_json:
#                 try:
#                     json_data = request.get_json()
#                     if json_data and 'message' in json_data:
#                         query_length = len(json_data['message'])
#                 except:
#                     pass
            
#             return response
            
#         except Exception as e:
#             status_code = 500
#             raise
        
#         finally:
#             processing_time = time.time() - start_time
            
#             # Création de la métrique
#             metric = RequestMetrics(
#                 timestamp=start_time,
#                 endpoint=endpoint,
#                 method=request.method,
#                 response_time=processing_time,
#                 status_code=status_code,
#                 user_agent=request.headers.get('User-Agent', ''),
#                 ip_address=request.remote_addr or '',
#                 query_length=query_length,
#                 response_length=response_length,
#                 territory_detected=territory_detected,
#                 indicator_detected=indicator_detected,
#                 search_method=search_method,
#                 confidence_score=confidence_score
#             )
            
#             metrics_collector.add_request(metric)
    
#     return decorated_function


# def _safe_call(obj, method_name: str, *args, default=None):
#     """Appelle une méthode de manière sécurisée."""
#     try:
#         if obj and hasattr(obj, method_name):
#             method = getattr(obj, method_name)
#             if callable(method):
#                 return method(*args)
#     except Exception as e:
#         logger.debug(f"Erreur appel sécurisé {method_name}: {e}")
#     return default


# def initialize_chatbot_app(config) -> bool:
#     """Initialise l'application chatbot avec gestion d'erreurs robuste."""
#     global _data_processor, _chatbot, _config
#     _config = config

#     try:
#         logger.info("Initialisation du système chatbot optimisé...")

#         # Import dynamique des modules
#         HCPDataProcessor = None
#         HCPChatbotOptimized = None
        
#         # Tentative d'import du processeur de données
#         for module_path in ['src.data_processor', 'data_processor']:
#             try:
#                 module = __import__(module_path, fromlist=['HCPDataProcessor'])
#                 HCPDataProcessor = getattr(module, 'HCPDataProcessor', None)
#                 if HCPDataProcessor:
#                     logger.info(f"HCPDataProcessor importé depuis {module_path}")
#                     break
#             except ImportError:
#                 continue
        
#         # Import du chatbot optimisé
#         for module_path in ['src.chatbot', 'chatbot']:
#             try:
#                 module = __import__(module_path, fromlist=['HCPChatbotOptimized'])
#                 HCPChatbotOptimized = getattr(module, 'HCPChatbotOptimized', None)
#                 if HCPChatbotOptimized:
#                     logger.info(f"HCPChatbotOptimized importé depuis {module_path}")
#                     break
#             except ImportError:
#                 continue
        
#         if not HCPChatbotOptimized:
#             logger.error("Impossible d'importer HCPChatbotOptimized")
#             return False

#         # Initialisation du processeur de données
#         if HCPDataProcessor:
#             try:
#                 logger.info("Initialisation du processeur de données...")
#                 _data_processor = HCPDataProcessor(config)
                
#                 # Chargement des données
#                 data = _safe_call(_data_processor, 'load_all_data')
#                 if data is not None:
#                     try:
#                         data_count = len(data) if hasattr(data, '__len__') else 1
#                         logger.info(f"Données chargées: {data_count} entrées")
#                     except:
#                         logger.info("Données chargées (taille indéterminée)")
#                 else:
#                     logger.warning("Aucune donnée chargée par le processeur")
                    
#             except Exception as e:
#                 logger.warning(f"Erreur initialisation processeur de données: {e}")
#                 _data_processor = None

#         # Création du chatbot
#         logger.info("Création du chatbot optimisé...")
#         try:
#             _chatbot = HCPChatbotOptimized(config, _data_processor)
#             logger.info("Chatbot créé avec succès")
#         except Exception as e:
#             logger.error(f"Erreur création chatbot: {e}")
#             return False

#         # Chargement du modèle
#         try:
#             _safe_call(_chatbot, 'load_model')
#             model_status = "Modèle entraîné" if getattr(_chatbot, 'is_trained', False) else "Mode recherche sémantique"
#             logger.info(f"Modèle initialisé: {model_status}")
#         except Exception as e:
#             logger.warning(f"Problème chargement modèle: {e}")

#         # Initialisation des paires Q&A et index
#         try:
#             _safe_call(_chatbot, 'initialize_qa_pairs')
            
#             # Récupération des statistiques
#             stats = _safe_call(_chatbot, 'get_statistics', default={}) or {}
#             system_stats = stats.get('system', {})
            
#             logger.info("Système chatbot initialisé avec succès:")
#             logger.info(f"  • Paires Q&A: {system_stats.get('qa_pairs_count', 0)}")
#             logger.info(f"  • Embeddings: {system_stats.get('embedding_count', 0)}")
#             logger.info(f"  • Vocabulaire: {system_stats.get('vocabulary_size', 0)} mots")
#             logger.info(f"  • Index de recherche: {system_stats.get('search_index_size', 0)} entrées")
#             logger.info(f"  • Territoires uniques: {system_stats.get('unique_territories', 0)}")
            
#         except Exception as e:
#             logger.error(f"Erreur initialisation index: {e}")

#         return True

#     except Exception as e:
#         logger.error(f"Erreur critique lors de l'initialisation: {e}")
#         traceback.print_exc()
#         return False


# # Configuration du logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger('FlaskAppOptimized')


# @app.route('/')
# @track_request_metrics
# def home():
#     """Page d'accueil avec métriques."""
#     try:
#         return render_template('index.html')
#     except Exception as e:
#         logger.error(f"Erreur rendu template: {e}")
#         return jsonify({
#             'error': 'Erreur chargement page d\'accueil',
#             'status': 'error'
#         }), 500


# @app.route('/chat', methods=['POST'])
# @track_request_metrics
# def chat_route():
#     """Endpoint de chat avec métriques enrichies et gestion d'erreurs avancée."""
#     global _chatbot
    
#     try:
#         # Validation du chatbot
#         if _chatbot is None:
#             logger.error("Tentative d'utilisation du chatbot non initialisé")
#             return jsonify({
#                 'error': 'Chatbot non initialisé',
#                 'status': 'error',
#                 'suggestion': 'Redémarrer l\'application'
#             }), 500

#         # Validation des données d'entrée
#         if not request.is_json:
#             return jsonify({
#                 'error': 'Content-Type doit être application/json',
#                 'status': 'error'
#             }), 400
        
#         data = request.get_json()
#         if not data or 'message' not in data:
#             return jsonify({
#                 'error': 'Message manquant dans la requête',
#                 'status': 'error'
#             }), 400

#         user_message = data['message'].strip()
#         if not user_message:
#             return jsonify({
#                 'error': 'Message vide',
#                 'status': 'error'
#             }), 400
        
#         if len(user_message) > 1000:  # Limite de sécurité
#             return jsonify({
#                 'error': 'Message trop long (max 1000 caractères)',
#                 'status': 'error'
#             }), 400

#         # Log de la requête
#         logger.info(f"Question: {user_message[:100]}{'...' if len(user_message) > 100 else ''}")

#         # Traitement de la requête
#         processing_start = time.time()
#         response = _safe_call(_chatbot, 'chat', user_message)
#         processing_time = time.time() - processing_start

#         if response is None:
#             logger.error("Chatbot n'a pas retourné de réponse")
#             return jsonify({
#                 'error': 'Aucune réponse générée',
#                 'status': 'error'
#             }), 500

#         # Collecte des métadonnées enrichies
#         metadata = {
#             # Informations de base
#             'processing_time_ms': round(processing_time * 1000, 2),
#             'query_length': len(user_message),
#             'response_length': len(response),
#             'timestamp': datetime.now().isoformat(),
            
#             # État du système
#             'model_used': 'trained' if getattr(_chatbot, 'is_trained', False) else 'search_only',
#             'has_embeddings': getattr(_chatbot, 'sentence_transformer', None) is not None,
#             'qa_pairs_count': len(getattr(_chatbot, 'qa_pairs', [])),
            
#             # Détection d'entités
#             'territory_detected': _safe_call(_chatbot, 'advanced_extract_territory', user_message),
#             'indicator_detected': _safe_call(_chatbot, 'advanced_extract_indicator', user_message),
#             'is_greeting': _safe_call(_chatbot, 'is_greeting', user_message, default=False),
            
#             # Méthode de recherche utilisée
#             'response_method': _determine_response_method(_chatbot, user_message),
#             'confidence_score': _estimate_confidence_score(_chatbot, user_message, response),
            
#             # Statistiques système
#             'system_stats': _safe_call(_chatbot, 'get_statistics', default={}),
#         }

#         # Log des résultats
#         logger.info(f"Réponse générée en {processing_time:.3f}s")
#         if metadata.get('territory_detected'):
#             logger.debug(f"Territoire: {metadata['territory_detected']}")
#         if metadata.get('indicator_detected'):
#             logger.debug(f"Indicateur: {metadata['indicator_detected'][0] if isinstance(metadata['indicator_detected'], tuple) else metadata['indicator_detected']}")

#         return jsonify({
#             'response': response,
#             'status': 'success',
#             'metadata': metadata
#         })

#     except Exception as e:
#         logger.error(f"Erreur traitement requête chat: {e}")
#         traceback.print_exc()
        
#         error_details = str(e) if _config and getattr(_config, 'DEBUG', False) else None
        
#         return jsonify({
#             'error': 'Erreur interne du serveur',
#             'status': 'error',
#             'details': error_details,
#             'suggestion': 'Vérifiez les logs pour plus de détails'
#         }), 500


# def _determine_response_method(chatbot, query: str) -> str:
#     """Détermine la méthode de réponse utilisée."""
#     if not chatbot:
#         return 'unknown'
    
#     # Vérification des salutations
#     if _safe_call(chatbot, 'is_greeting', query, default=False):
#         return 'greeting'
    
#     # Vérification du modèle entraîné
#     if getattr(chatbot, 'is_trained', False) and getattr(chatbot, 'model', None):
#         return 'trained_model'
    
#     # Vérification de la recherche sémantique
#     if getattr(chatbot, 'sentence_transformer', None):
#         return 'semantic_search'
    
#     # Vérification de l'index de recherche
#     if getattr(chatbot, 'search_index', None):
#         return 'index_search'
    
#     return 'text_matching'


# def _estimate_confidence_score(chatbot, query: str, response: str) -> float:
#     """Estime un score de confiance pour la réponse."""
#     try:
#         confidence = 0.5  # Score de base
        
#         # Bonus pour détection d'entités
#         if _safe_call(chatbot, 'advanced_extract_territory', query):
#             confidence += 0.15
        
#         indicator_result = _safe_call(chatbot, 'advanced_extract_indicator', query)
#         if indicator_result and (indicator_result[0] if isinstance(indicator_result, tuple) else indicator_result):
#             confidence += 0.15
        
#         # Pénalité pour réponses par défaut
#         default_responses = getattr(chatbot, 'default_responses', [])
#         if any(default in response for default in default_responses):
#             confidence -= 0.3
        
#         # Bonus/pénalité selon la longueur de la réponse
#         if 30 <= len(response) <= 500:
#             confidence += 0.1
#         elif len(response) < 30:
#             confidence -= 0.2
        
#         # Bonus pour modèle entraîné
#         if getattr(chatbot, 'is_trained', False):
#             confidence += 0.1
        
#         return max(0.0, min(1.0, confidence))
        
#     except Exception:
#         return 0.5


# @app.route('/health')
# @track_request_metrics
# def health():
#     """Endpoint de santé avec diagnostic complet."""
#     try:
#         health_data = {
#             'status': 'healthy',
#             'timestamp': datetime.now().isoformat(),
#             'system_info': {
#                 'chatbot_initialized': _chatbot is not None,
#                 'data_processor_initialized': _data_processor is not None,
#             }
#         }

#         # Test du chatbot
#         if _chatbot:
#             try:
#                 test_response = _safe_call(_chatbot, 'chat', 'Test santé système')
#                 health_data['chatbot_test'] = {
#                     'passed': bool(test_response),
#                     'response_length': len(test_response) if test_response else 0
#                 }
#             except Exception as e:
#                 health_data['chatbot_test'] = {
#                     'passed': False,
#                     'error': str(e)
#                 }

#         # Statistiques système
#         if _chatbot:
#             stats = _safe_call(_chatbot, 'get_statistics', default={})
#             health_data['system_stats'] = stats

#         # Métriques de performance
#         metrics_report = metrics_collector.get_comprehensive_report()
#         health_data['performance_metrics'] = metrics_report
        
#         # Détermination du statut global
#         critical_issues = []
        
#         if not _chatbot:
#             critical_issues.append("Chatbot non initialisé")
#         elif not getattr(_chatbot, 'qa_pairs', []):
#             critical_issues.append("Aucune paire Q&A disponible")
        
#         system_health = metrics_report.get('system_health', {})
#         if system_health.get('status') in ['critical', 'warning']:
#             critical_issues.append(f"Performance système: {system_health.get('status')}")
        
#         if critical_issues:
#             health_data['status'] = 'unhealthy'
#             health_data['critical_issues'] = critical_issues
        
#         status_code = 200 if health_data['status'] == 'healthy' else 503
#         return jsonify(health_data), status_code

#     except Exception as e:
#         logger.error(f"Erreur endpoint health: {e}")
#         return jsonify({
#             'status': 'error',
#             'error': str(e),
#             'timestamp': datetime.now().isoformat()
#         }), 500


# @app.route('/metrics')
# @track_request_metrics
# def metrics_route():
#     """Endpoint pour métriques détaillées."""
#     try:
#         comprehensive_report = metrics_collector.get_comprehensive_report()
        
#         # Ajout des métriques spécifiques au chatbot
#         chatbot_metrics = {}
#         if _chatbot:
#             chatbot_metrics = {
#                 'statistics': _safe_call(_chatbot, 'get_statistics', default={}),
#                 'performance_report': _safe_call(_chatbot, 'get_performance_report', default={})
#             }
        
#         return jsonify({
#             'status': 'success',
#             'timestamp': datetime.now().isoformat(),
#             'application_metrics': comprehensive_report,
#             'chatbot_metrics': chatbot_metrics
#         })
        
#     except Exception as e:
#         logger.error(f"Erreur endpoint metrics: {e}")
#         return jsonify({
#             'error': str(e),
#             'status': 'error'
#         }), 500


# @app.route('/territories')
# @track_request_metrics
# def territories_route():
#     """Endpoint pour récupérer les territoires et indicateurs disponibles."""
#     try:
#         territories_set = set()
#         indicators_set = set()
#         genres_set = set()
#         sources_set = set()

#         # Collecte depuis les données
#         data_sources = []
        
#         if _data_processor and hasattr(_data_processor, 'combined_data'):
#             data_sources.append(('data_processor', _data_processor.combined_data))
        
#         if _chatbot and hasattr(_chatbot, 'qa_pairs'):
#             data_sources.append(('chatbot', _chatbot.qa_pairs))

#         for source_name, data in data_sources:
#             if not data:
#                 continue
                
#             for item in data:
#                 # Territoire
#                 territory = (
#                     item.get('territory') or 
#                     item.get('original_territory') or 
#                     item.get('territoire') or ''
#                 ).strip()
#                 if territory and territory != 'Unknown':
#                     territories_set.add(territory)
                
#                 # Indicateur
#                 indicator = (
#                     item.get('variable') or 
#                     item.get('indicateur') or 
#                     item.get('column_original') or ''
#                 ).strip()
#                 if indicator and indicator != 'unknown':
#                     indicators_set.add(indicator)
                
#                 # Genre
#                 genre = (item.get('sexe') or item.get('genre') or '').strip()
#                 if genre:
#                     genres_set.add(genre)
                
#                 # Source
#                 source = (
#                     item.get('source_data') or 
#                     item.get('source') or ''
#                 ).strip()
#                 if source and source != 'non spécifié':
#                     sources_set.add(source)

#         # Tri et limitation
#         territories = sorted(territories_set)
#         indicators = sorted(indicators_set)
#         genres = sorted(genres_set)
#         sources = sorted(sources_set)

#         # Ajout des valeurs par défaut
#         default_entries = {
#             'territories': ['Ensemble du territoire national', 'Maroc'],
#             'genres': ['ensemble']
#         }
        
#         for key, defaults in default_entries.items():
#             current_list = locals()[key]
#             for default in defaults:
#                 if default not in current_list:
#                     current_list.insert(0, default)

#         result = {
#             'territories': territories[:100],  # Limiter pour performance
#             'indicators': indicators[:50],
#             'genres': genres,
#             'sources': sources[:30],
#             'status': 'success',
#             'metadata': {
#                 'total_counts': {
#                     'territories': len(territories_set),
#                     'indicators': len(indicators_set),
#                     'genres': len(genres_set),
#                     'sources': len(sources_set)
#                 },
#                 'data_sources_used': [name for name, data in data_sources if data],
#                 'timestamp': datetime.now().isoformat()
#             }
#         }

#         return jsonify(result)

#     except Exception as e:
#         logger.error(f"Erreur endpoint territories: {e}")
#         traceback.print_exc()
#         return jsonify({
#             'error': str(e),
#             'status': 'error'
#         }), 500


# # Gestionnaires d'erreur
# @app.errorhandler(404)
# def not_found(error):
#     return jsonify({
#         'error': 'Endpoint non trouvé',
#         'status': 'error',
#         'available_endpoints': ['/chat', '/health', '/metrics', '/territories']
#     }), 404


# @app.errorhandler(500)
# def internal_error(error):
#     logger.error(f"Erreur 500: {error}")
#     return jsonify({
#         'error': 'Erreur interne du serveur',
#         'status': 'error'
#     }), 500


# @app.errorhandler(413)
# def payload_too_large(error):
#     return jsonify({
#         'error': 'Requête trop volumineuse',
#         'status': 'error'
#     }), 413


# def create_app_with_config(config):
#     """Crée l'application Flask avec configuration."""
#     global _config
#     _config = config
    
#     # Configuration Flask
#     if hasattr(config, 'SECRET_KEY'):
#         app.secret_key = config.SECRET_KEY
    
#     # Configuration des limites
#     app.config['MAX_CONTENT_LENGTH'] = getattr(config, 'MAX_CONTENT_LENGTH', 1 * 1024 * 1024)  # 1MB
    
#     # Initialisation du chatbot
#     success = initialize_chatbot_app(config)
    
#     if success:
#         logger.info("Application Flask initialisée avec succès")
#     else:
#         logger.error("Échec de l'initialisation - fonctionnalités limitées")
    
#     return app


# # Point d'entrée principal
# if __name__ == '__main__':
#     try:
#         from config import Config
#         config = Config()
        
#         create_app_with_config(config)
        
#         # Configuration serveur
#         host = getattr(config, 'FLASK_HOST', '0.0.0.0')
#         port = getattr(config, 'FLASK_PORT', 5000)
#         debug = getattr(config, 'FLASK_DEBUG', False)
        
#         logger.info(f"Démarrage serveur Flask - {host}:{port} (debug={debug})")
        
#         app.run(
#             host=host, 
#             port=port, 
#             debug=debug, 
#             use_reloader=False,  # Éviter les conflits avec les métriques
#             threaded=True
#         )
        
#     except Exception as e:
#         logger.error(f"Erreur critique au démarrage: {e}")
#         traceback.print_exc()





