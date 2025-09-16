from flask import Blueprint, request, jsonify
from src.chatbot import HCPChatbot
from src.data_processor import HCPDataProcessor
from config import Config
import logging

logger = logging.getLogger(__name__)
api = Blueprint('api', __name__)

# Initialisation du processor et du chatbot
_config = Config()
data_processor = HCPDataProcessor(_config)
chatbot = HCPChatbot(_config, data_processor)

# Charger et initialiser avant les requêtes
@api.before_app_first_request
def initialize():
    try:
        logger.info("Chargement des données indicators.json...")
        data_processor.load_data(_config.DATA_PATH)
        chatbot.initialize_qa_pairs()
        chatbot.load_model()
        logger.info("API HCP Agadir initialisée avec succès")
    except Exception as e:
        logger.error(f"Erreur d'initialisation API : {e}")

@api.route('/chat', methods=['POST'])
def chat():
    payload = request.get_json() or {}
    user_msg = payload.get('message', '').strip()
    if not user_msg:
        return jsonify({'error': 'Message vide'}), 400
    try:
        response = chatbot.chat(user_msg)
        return jsonify({'response': response, 'status': 'success'})
    except Exception as e:
        logger.error(f"Erreur lors du chat: {e}")
        return jsonify({'error': 'Erreur interne du serveur'}), 500

@api.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': chatbot.model is not None,
        'data_loaded': data_processor.data is not None
    })
