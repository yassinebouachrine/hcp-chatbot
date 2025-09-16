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
