# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import re
# import json
# from typing import List, Dict, Optional, Tuple
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import random
# from difflib import SequenceMatcher
# import logging
# import os
# import time

# class HCPChatbot:
#     def __init__(self, config, data_processor):
#         self.config = config
#         self.data_processor = data_processor
#         self.tokenizer = None
#         self.model = None
#         self.qa_pairs = []
#         self.is_trained = False
#         self.sentence_transformer = None
        
#         # Configuration du logging
#         logging.basicConfig(level=getattr(config, 'LOG_LEVEL', logging.INFO))
#         self.logger = logging.getLogger(__name__)
        
#         # Réponses par défaut améliorées
#         self.default_responses = [
#             "Je ne trouve pas d'informations précises sur cette question dans ma base de données HCP. Pouvez-vous reformuler en précisant le territoire et le type de statistique recherché ?",
#             "Cette information spécifique n'est pas disponible dans mes données. Je peux vous aider avec la population légale, municipale, ou les tranches d'âge pour différents territoires du Maroc.",
#             "Je n'ai pas de données exactes correspondant à votre question. Essayez de mentionner un territoire précis (ex: 'Ensemble du territoire national') et un indicateur spécifique."
#         ]
        
#         self.greeting_responses = [
#             "Bonjour ! Je suis l'assistant statistique du HCP. Je peux vous fournir des données précises sur la population légale, municipale et les tranches d'âge au Maroc. Posez-moi une question spécifique !",
#             "Salut ! Je dispose de statistiques démographiques détaillées du Maroc. Demandez-moi par exemple : 'Quelle est la population légale du Maroc ?' ou 'Quel est le pourcentage de 0-4 ans au niveau national ?'",
#             "Bienvenue ! Je suis spécialisé dans les données HCP du Maroc. Je connais la population légale, municipale et la répartition par âge. Comment puis-je vous aider ?"
#         ]

#     def load_sentence_transformer(self):
#         """Charge le modèle de sentence transformer avec vérifications"""
#         try:
#             model_name = getattr(self.config, 'EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
#             self.sentence_transformer = SentenceTransformer(model_name)
#             self.logger.info(f"✅ Sentence Transformer {model_name} chargé avec succès")
#         except Exception as e:
#             self.logger.error(f"❌ Erreur lors du chargement du Sentence Transformer: {e}")
#             self.sentence_transformer = None
    
#     def load_model(self, model_path: str = None):
#         """Charge le modèle avec priorité sur la recherche sémantique"""
#         path = model_path or self.config.MODEL_PATH
        
#         try:
#             # Vérifier si le modèle personnalisé existe
#             if os.path.exists(path) and os.path.exists(os.path.join(path, 'config.json')):
#                 self.logger.info(f"🔄 Chargement du modèle personnalisé depuis {path}")
                
#                 self.tokenizer = AutoTokenizer.from_pretrained(path)
#                 if self.tokenizer.pad_token is None:
#                     self.tokenizer.pad_token = self.tokenizer.eos_token
                
#                 self.model = AutoModelForCausalLM.from_pretrained(
#                     path,
#                     torch_dtype=torch.float32,
#                     low_cpu_mem_usage=True
#                 )
                
#                 self.is_trained = True
#                 self.logger.info("✅ Modèle personnalisé chargé avec succès")
                
#             else:
#                 self.logger.warning(f"⚠️ Modèle personnalisé non trouvé à {path}")
#                 self.logger.info("📋 Mode recherche sémantique uniquement activé")
#                 self.is_trained = False
                
#         except Exception as e:
#             self.logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
#             self.logger.info("📋 Utilisation de la recherche sémantique uniquement")
#             self.is_trained = False
    
#     def preprocess_query(self, query: str) -> str:
#         """Préprocesse et normalise la requête utilisateur"""
#         if not query:
#             return ""
        
#         query = query.strip()
        
#         # Normalisations spécifiques - CORRECTION DE LA SYNTAXE
#         replacements = {
#             r'\bmaroc\b': 'Ensemble du territoire national',
#             r'\broyaume du maroc\b': 'Ensemble du territoire national',
#             r'\bterritoire national\b': 'Ensemble du territoire national',
#             r'\bnational\b': 'Ensemble du territoire national',
#             r'\bcombien d\'habitants\b': 'population légale',
#             r'\bcombien de personnes\b': 'population légale',
#             r'\bpopulation totale\b': 'population légale',
#             r'\bnombre d\'habitants\b': 'population légale'
#         }
        
#         query_processed = query.lower()
#         for pattern, replacement in replacements.items():
#             query_processed = re.sub(pattern, replacement, query_processed, flags=re.IGNORECASE)
        
#         return query_processed
    
#     def extract_territory_from_query(self, query: str) -> Optional[str]:
#         """Extrait et normalise le territoire de la requête"""
#         query_lower = query.lower()
        
#         # Territoires du Maroc avec priorités
#         territory_patterns = [
#             (r'\bensemble du territoire national\b', 'Ensemble du territoire national'),
#             (r'\bterritoire national\b', 'Ensemble du territoire national'),
#             (r'\bmaroc\b', 'Ensemble du territoire national'),
#             (r'\broyaume du maroc\b', 'Ensemble du territoire national'),
#             (r'\bnational\b', 'Ensemble du territoire national'),
#         ]
        
#         for pattern, standard_name in territory_patterns:
#             if re.search(pattern, query_lower):
#                 return standard_name
        
#         # Chercher dans les territoires des données
#         if hasattr(self.data_processor, 'combined_data') and self.data_processor.combined_data:
#             territories = set([item.get('territoire', '') for item in self.data_processor.combined_data[:1000]])
            
#             for territory in territories:
#                 if territory and territory.lower() in query_lower:
#                     return territory
        
#         return None
    
#     def extract_indicator_from_query(self, query: str) -> Optional[str]:
#         """Extrait l'indicateur démographique de la requête"""
#         query_lower = query.lower()
        
#         # Patterns d'indicateurs spécifiques
#         indicator_patterns = {
#             'population_legale': [r'population légale', r'population legale', r'pop légale'],
#             'population_municipale': [r'population municipale', r'pop municipale'],
#             'age_0-4 ans': [r'0-4 ans', r'0 à 4 ans', r'moins de 5 ans'],
#             'age_5-9 ans': [r'5-9 ans', r'5 à 9 ans'],
#             'age_10-14 ans': [r'10-14 ans', r'10 à 14 ans'],
#             'age_15-19 ans': [r'15-19 ans', r'15 à 19 ans'],
#             'age_20-24 ans': [r'20-24 ans', r'20 à 24 ans'],
#             'age_25-29 ans': [r'25-29 ans', r'25 à 29 ans'],
#             'age_30-34 ans': [r'30-34 ans', r'30 à 34 ans'],
#             'age_35-39 ans': [r'35-39 ans', r'35 à 39 ans'],
#         }
        
#         for indicator, patterns in indicator_patterns.items():
#             for pattern in patterns:
#                 if re.search(pattern, query_lower):
#                     return indicator
        
#         # Patterns généraux
#         if re.search(r'pourcentage|%|taux', query_lower):
#             if re.search(r'\d+-\d+\s*ans', query_lower):
#                 age_match = re.search(r'(\d+-\d+)\s*ans', query_lower)
#                 if age_match:
#                     return f"age_{age_match.group(1)} ans"
        
#         return None
    
#     def is_greeting(self, query: str) -> bool:
#         """Détection améliorée des salutations"""
#         greetings = [
#             r'\bbonjour\b', r'\bbonsoir\b', r'\bsalut\b', r'\bhello\b', r'\bhi\b',
#             r'\baide\b', r'\baider\b', r'\bmerci\b', r'\bau revoir\b',
#             r'^(salut|hello|hi|bonjour|bonsoir)$'
#         ]
        
#         query_lower = query.lower().strip()
#         return any(re.search(pattern, query_lower) for pattern in greetings)
    
#     def find_best_match_semantic(self, query: str) -> Optional[Dict]:
#         """Recherche sémantique améliorée avec scoring multiple"""
#         if not self.sentence_transformer or not self.qa_pairs:
#             return None
        
#         try:
#             processed_query = self.preprocess_query(query)
#             territory = self.extract_territory_from_query(query)
#             indicator = self.extract_indicator_from_query(query)
            
#             # Créer une requête enrichie
#             enriched_query_parts = [processed_query]
#             if territory:
#                 enriched_query_parts.append(f"territoire: {territory}")
#             if indicator:
#                 enriched_query_parts.append(f"indicateur: {indicator}")
            
#             enriched_query = " | ".join(enriched_query_parts)
#             query_embedding = self.sentence_transformer.encode([enriched_query])
            
#             best_score = 0
#             best_match = None
            
#             for qa_pair in self.qa_pairs:
#                 if 'embedding' not in qa_pair:
#                     continue
                
#                 # Similarité sémantique
#                 similarity = np.dot(query_embedding[0], qa_pair['embedding'])
#                 score = float(similarity)
                
#                 # Bonus pour correspondance exacte de territoire
#                 if territory and qa_pair.get('territoire', '').lower() == territory.lower():
#                     score += 0.3
                
#                 # Bonus pour correspondance d'indicateur
#                 if indicator and qa_pair.get('indicateur', '').lower() == indicator.lower():
#                     score += 0.2
                
#                 # Bonus pour genre si pertinent
#                 if 'ensemble' in query.lower() and qa_pair.get('genre', '') == 'ensemble':
#                     score += 0.1
                
#                 if score > best_score:
#                     best_score = score
#                     best_match = qa_pair
            
#             # Seuil de confiance adaptatif
#             threshold = getattr(self.config, 'SIMILARITY_THRESHOLD', 0.75)
#             if best_score > threshold:
#                 self.logger.info(f"✅ Correspondance trouvée (score: {best_score:.3f})")
#                 return best_match
#             else:
#                 self.logger.info(f"⚠️ Aucune correspondance suffisante (meilleur score: {best_score:.3f})")
#                 return None
                
#         except Exception as e:
#             self.logger.error(f"❌ Erreur dans la recherche sémantique: {e}")
#             return None
    
#     def find_best_match_textual(self, query: str) -> Optional[Dict]:
#         """Recherche textuelle de fallback améliorée"""
#         if not self.qa_pairs:
#             return None
        
#         processed_query = self.preprocess_query(query)
#         territory = self.extract_territory_from_query(query)
#         indicator = self.extract_indicator_from_query(query)
        
#         best_score = 0
#         best_match = None
        
#         for qa_pair in self.qa_pairs:
#             question_text = qa_pair.get('question', '').lower()
            
#             # Score de similarité textuelle
#             score = SequenceMatcher(None, processed_query, question_text).ratio()
            
#             # Bonus pour correspondances exactes
#             if territory and qa_pair.get('territoire', '').lower() == territory.lower():
#                 score += 0.4
#             if indicator and qa_pair.get('indicateur', '').lower() == indicator.lower():
#                 score += 0.3
            
#             # Bonus pour mots-clés importants
#             important_keywords = ['population', 'pourcentage', 'légale', 'municipale']
#             for keyword in important_keywords:
#                 if keyword in processed_query and keyword in question_text:
#                     score += 0.1
            
#             if score > best_score and score > 0.6:
#                 best_score = score
#                 best_match = qa_pair
        
#         return best_match if best_score > 0.6 else None
    
#     def search_by_filters(self, territory: str = None, question_type: str = None, indicateur: str = None) -> Optional[Dict]:
#         """Recherche par filtres exacts"""
#         if not self.qa_pairs:
#             return None
        
#         candidates = []
        
#         for qa_pair in self.qa_pairs:
#             match = True
            
#             # Filtrer par territoire
#             if territory:
#                 qa_territory = qa_pair.get('territoire', qa_pair.get('territory', ''))
#                 if qa_territory.lower() != territory.lower():
#                     match = False
            
#             # Filtrer par indicateur
#             if indicateur:
#                 qa_indicator = qa_pair.get('indicateur', qa_pair.get('variable', ''))
#                 if qa_indicator.lower() != indicateur.lower():
#                     match = False
            
#             # Filtrer par type de question
#             if question_type:
#                 if qa_pair.get('question_type', '') != question_type:
#                     match = False
            
#             if match:
#                 candidates.append(qa_pair)
        
#         # Retourner le premier candidat ou celui avec le meilleur score
#         if candidates:
#             # Prioriser 'ensemble' comme genre par défaut
#             ensemble_candidates = [c for c in candidates if c.get('genre', '') == 'ensemble']
#             return ensemble_candidates[0] if ensemble_candidates else candidates[0]
        
#         return None
    
#     def generate_smart_response(self, query: str) -> str:
#         """Génération de réponse avec stratégie intelligente"""
#         if self.is_greeting(query):
#             return random.choice(self.greeting_responses)
        
#         # 1. Recherche sémantique (priorité)
#         if self.sentence_transformer:
#             semantic_match = self.find_best_match_semantic(query)
#             if semantic_match:
#                 return semantic_match['answer']
        
#         # 2. Recherche par filtres exacts
#         territory = self.extract_territory_from_query(query)
#         indicator = self.extract_indicator_from_query(query)
        
#         if territory or indicator:
#             filtered_result = self.search_by_filters(territory, None, indicator)
#             if filtered_result:
#                 return filtered_result['answer']
        
#         # 3. Recherche textuelle de fallback
#         textual_match = self.find_best_match_textual(query)
#         if textual_match:
#             return textual_match['answer']
        
#         # 4. Réponse guidée si aucune correspondance
#         guidance = self.generate_guidance_response(query, territory, indicator)
#         return guidance
    
#     def generate_guidance_response(self, query: str, territory: str = None, indicator: str = None) -> str:
#         """Génère une réponse d'aide contextuelle"""
#         guidance_parts = []
        
#         if territory:
#             guidance_parts.append(f"Territoire détecté: {territory}")
#         if indicator:
#             guidance_parts.append(f"Indicateur détecté: {indicator}")
        
#         # Suggestions contextuelles
#         suggestions = []
        
#         if 'population' in query.lower():
#             suggestions.extend([
#                 "Quelle est la population légale de l'ensemble du territoire national ?",
#                 "Population municipale du Maroc ?"
#             ])
        
#         if 'pourcentage' in query.lower() or '%' in query:
#             suggestions.extend([
#                 "Quel est le pourcentage de 0-4 ans au niveau national ?",
#                 "Pourcentage de la population de 15-19 ans ?"
#             ])
        
#         if not suggestions:
#             suggestions = [
#                 "Quelle est la population légale du Maroc ?",
#                 "Quel est le pourcentage de 20-24 ans au niveau national ?",
#                 "Population municipale de l'ensemble du territoire national ?"
#             ]
        
#         response_parts = [
#             "Je n'ai pas trouvé de correspondance exacte pour votre question."
#         ]
        
#         if guidance_parts:
#             response_parts.append(" ".join(guidance_parts))
        
#         response_parts.extend([
#             "\n\nVoici des exemples de questions que je peux traiter :",
#             *[f"• {suggestion}" for suggestion in suggestions[:3]]
#         ])
        
#         return " ".join(response_parts)
    
#     def chat(self, query: str) -> str:
#         """Interface principale du chatbot - version corrigée"""
#         if not query or not query.strip():
#             return "Veuillez poser une question sur les statistiques démographiques du Maroc."
        
#         try:
#             # Log de la requête
#             self.logger.info(f"🔍 Question reçue: {query}")
            
#             # Générer la réponse avec la stratégie intelligente
#             response = self.generate_smart_response(query)
            
#             # Valider et nettoyer la réponse
#             cleaned_response = self.clean_response(response)
            
#             # Log de la réponse
#             self.logger.info(f"📝 Réponse générée: {cleaned_response[:100]}...")
            
#             # Sauvegarder l'historique
#             self.save_conversation_history(query, cleaned_response)
            
#             return cleaned_response
            
#         except Exception as e:
#             self.logger.error(f"❌ Erreur dans chat: {e}")
#             return "Je suis désolé, une erreur s'est produite. Veuillez reformuler votre question sur les données démographiques du Maroc."
    
#     def clean_response(self, response: str) -> str:
#         """Nettoie et valide la réponse"""
#         if not response:
#             return random.choice(self.default_responses)
        
#         # Supprimer les tokens spéciaux
#         cleaned = re.sub(r'<\|[^|]+\|>', '', response)
#         cleaned = re.sub(r'\[.*?\]', '', cleaned)
        
#         # Supprimer les répétitions
#         cleaned = re.sub(r'(.{10,}?)\1+', r'\1', cleaned)
        
#         # Nettoyer les espaces
#         cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
#         # Vérifier la longueur minimale
#         if len(cleaned) < 20:
#             return random.choice(self.default_responses)
        
#         return cleaned
    
#     def initialize_qa_pairs(self):
#         """Initialisation améliorée des paires Q&A"""
#         try:
#             self.logger.info("🔄 Initialisation des paires question-réponse...")
            
#             # Récupérer les données du processeur
#             generated_pairs = []
            
#             if hasattr(self.data_processor, 'combined_data') and self.data_processor.combined_data:
#                 self.logger.info(f"📊 Traitement de {len(self.data_processor.combined_data)} éléments de données")
                
#                 for item in self.data_processor.combined_data:
#                     if item.get('question') and item.get('answer'):
#                         qa_pair = {
#                             'question': item.get('question', ''),
#                             'answer': item.get('answer', ''),
#                             'territoire': item.get('territoire', item.get('territory', 'Unknown')),
#                             'indicateur': item.get('indicateur', item.get('variable', 'unknown')),
#                             'genre': item.get('genre', item.get('sexe', 'ensemble')),
#                             'question_type': item.get('question_type', 'demographic')
#                         }
#                         generated_pairs.append(qa_pair)
            
#             # FAQ statiques
#             static_faqs = [
#                 {
#                     'question': "Qu'est-ce que le HCP ?",
#                     'answer': "Le Haut-Commissariat au Plan (HCP) est l'institution nationale marocaine chargée de la production et de la diffusion des statistiques officielles.",
#                     'territoire': 'Maroc',
#                     'indicateur': 'information_hcp',
#                     'genre': 'ensemble',
#                     'question_type': 'hcp'
#                 }
#             ]
            
#             # Fusionner les données
#             self.qa_pairs = static_faqs + generated_pairs
            
#             self.logger.info(f"✅ {len(self.qa_pairs)} paires Q&A chargées")
            
#             # Charger le sentence transformer et créer les embeddings
#             self.load_sentence_transformer()
            
#             if self.sentence_transformer and self.qa_pairs:
#                 self.logger.info("🔄 Création des embeddings sémantiques...")
                
#                 try:
#                     # Créer des textes enrichis pour les embeddings
#                     enriched_texts = []
#                     for pair in self.qa_pairs:
#                         text_parts = [pair['question']]
                        
#                         territoire = pair.get('territoire', '')
#                         indicateur = pair.get('indicateur', '')
                        
#                         if territoire and territoire != 'Unknown':
#                             text_parts.append(f"territoire: {territoire}")
#                         if indicateur and indicateur != 'unknown':
#                             text_parts.append(f"indicateur: {indicateur}")
                        
#                         enriched_text = " | ".join(text_parts)
#                         enriched_texts.append(enriched_text)
                    
#                     # Générer les embeddings
#                     embeddings = self.sentence_transformer.encode(enriched_texts, show_progress_bar=True)
                    
#                     # Assigner les embeddings aux paires
#                     for i, embedding in enumerate(embeddings):
#                         self.qa_pairs[i]['embedding'] = embedding
                    
#                     self.logger.info(f"✅ {len(embeddings)} embeddings créés avec succès")
                    
#                 except Exception as e:
#                     self.logger.error(f"❌ Erreur lors de la création des embeddings: {e}")
            
#             # Statistiques finales
#             territories = set(pair.get('territoire', 'Unknown') for pair in self.qa_pairs)
#             indicators = set(pair.get('indicateur', 'unknown') for pair in self.qa_pairs)
            
#             self.logger.info(f"📊 Couverture:")
#             self.logger.info(f"   • Territoires: {len(territories)}")
#             self.logger.info(f"   • Indicateurs: {len(indicators)}")
            
#         except Exception as e:
#             self.logger.error(f"❌ Erreur lors de l'initialisation des paires Q&A: {e}")
#             self.qa_pairs = []
    
#     def save_conversation_history(self, query: str, response: str):
#         """Sauvegarde l'historique avec métadonnées"""
#         if not getattr(self.config, 'SAVE_CONVERSATION_HISTORY', False):
#             return
            
#         try:
#             history_path = getattr(self.config, 'CONVERSATION_HISTORY_PATH', 'data/conversation_history.json')
            
#             # Charger l'historique existant
#             history = []
#             if os.path.exists(history_path):
#                 with open(history_path, 'r', encoding='utf-8') as f:
#                     history = json.load(f)
            
#             # Ajouter la nouvelle conversation avec métadonnées
#             conversation_entry = {
#                 'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
#                 'query': query,
#                 'response': response,
#                 'method': 'semantic_search' if self.sentence_transformer else 'textual_search',
#                 'territory_detected': self.extract_territory_from_query(query),
#                 'indicator_detected': self.extract_indicator_from_query(query)
#             }
            
#             history.append(conversation_entry)
            
#             # Garder seulement les 1000 dernières
#             if len(history) > 1000:
#                 history = history[-1000:]
            
#             # Créer le dossier si nécessaire
#             os.makedirs(os.path.dirname(history_path), exist_ok=True)
            
#             # Sauvegarder
#             with open(history_path, 'w', encoding='utf-8') as f:
#                 json.dump(history, f, indent=2, ensure_ascii=False)
                
#         except Exception as e:
#             self.logger.error(f"❌ Erreur lors de la sauvegarde: {e}")
    
#     def get_statistics(self) -> Dict:
#         """Statistiques complètes du chatbot"""
#         stats = {
#             'model_loaded': self.model is not None,
#             'is_trained': self.is_trained,
#             'qa_pairs_count': len(self.qa_pairs),
#             'has_sentence_transformer': self.sentence_transformer is not None,
#             'embedding_count': sum(1 for pair in self.qa_pairs if 'embedding' in pair),
#             'base_model': getattr(self.config, 'BASE_MODEL', 'Unknown'),
#             'data_structure': 'nouvelle_structure_qa_pairs_optimisée'
#         }
        
#         if self.qa_pairs:
#             territories = set(pair.get('territoire', 'Unknown') for pair in self.qa_pairs)
#             indicators = set(pair.get('indicateur', 'unknown') for pair in self.qa_pairs)
#             genres = set(pair.get('genre', 'ensemble') for pair in self.qa_pairs)
            
#             stats.update({
#                 'unique_territories': len(territories),
#                 'unique_indicators': len(indicators),
#                 'unique_genres': len(genres),
#                 'coverage': {
#                     'territories_sample': sorted(list(territories))[:10],
#                     'indicators_sample': sorted(list(indicators))[:10],
#                     'genres': sorted(list(genres))
#                 }
#             })
        
#         return stats
    
#     def get_help_message(self) -> str:
#         """Message d'aide détaillé"""
#         help_msg = """🤖 Assistant HCP - Guide d'utilisation

# Je suis spécialisé dans les statistiques démographiques du Maroc du Haut-Commissariat au Plan.

# ✅ Types de questions supportées:
# • Population légale et municipale
# • Répartition par tranches d'âge
# • Pourcentages démographiques
# • Informations sur le HCP

# 📝 Exemples de questions efficaces:
# • "Quelle est la population légale du Maroc ?"
# • "Population municipale de l'ensemble du territoire national ?"
# • "Quel est le pourcentage de 0-4 ans au niveau national ?"
# • "Pourcentage de la population de 25-29 ans ?"

# 💡 Conseils pour de meilleures réponses:
# • Mentionnez explicitement le territoire (ex: "Maroc", "ensemble du territoire national")
# • Soyez précis sur l'indicateur recherché (population légale/municipale, tranche d'âge)
# • Utilisez des questions complètes plutôt que des mots-clés

# 🔍 Recherche intelligente:
# Le système utilise la recherche sémantique pour comprendre vos questions même si elles ne correspondent pas exactement aux données."""
        
#         return help_msg


# def create_chatbot_with_config(config, data_processor):
#     """Crée et initialise le chatbot corrigé"""
#     chatbot = HCPChatbot(config, data_processor)
#     chatbot.load_model()
#     chatbot.initialize_qa_pairs()
#     return chatbot











# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import re
# import json
# from typing import List, Dict, Optional, Tuple
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import random
# from difflib import SequenceMatcher
# import logging
# import os
# import time

# class HCPChatbot:
#     def __init__(self, config, data_processor):
#         self.config = config
#         self.data_processor = data_processor
#         self.tokenizer = None
#         self.model = None
#         self.qa_pairs = []
#         self.is_trained = False
#         self.sentence_transformer = None
        
#         # Configuration du logging
#         logging.basicConfig(level=getattr(config, 'LOG_LEVEL', logging.INFO))
#         self.logger = logging.getLogger(__name__)
        
#         # Réponses par défaut améliorées
#         self.default_responses = [
#             "Je ne trouve pas d'informations précises sur cette question dans ma base de données HCP. Pouvez-vous reformuler en précisant le territoire, le type de statistique recherché et si vous cherchez des données de population ou de ménages ?",
#             "Cette information spécifique n'est pas disponible dans mes données. Je peux vous aider avec la population légale, municipale, les tranches d'âge ou les données sur les ménages pour différents territoires du Maroc.",
#             "Je n'ai pas de données exactes correspondant à votre question. Essayez de mentionner un territoire précis (ex: 'Ensemble du territoire national'), un indicateur spécifique et précisez s'il s'agit de données démographiques ou de ménages."
#         ]
        
#         self.greeting_responses = [
#             "Bonjour ! Je suis l'assistant statistique du HCP. Je peux vous fournir des données précises sur la population légale, municipale, les tranches d'âge et les statistiques des ménages au Maroc. Posez-moi une question spécifique !",
#             "Salut ! Je dispose de statistiques démographiques et de ménages détaillées du Maroc. Demandez-moi par exemple : 'Quelle est la population légale du Maroc ?' ou 'Statistiques des ménages au niveau national ?'",
#             "Bienvenue ! Je suis spécialisé dans les données HCP du Maroc. Je connais la population légale, municipale, la répartition par âge et les données sur les ménages. Comment puis-je vous aider ?"
#         ]

#     def load_sentence_transformer(self):
#         """Charge le modèle de sentence transformer avec vérifications"""
#         try:
#             model_name = getattr(self.config, 'EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
#             self.sentence_transformer = SentenceTransformer(model_name)
#             self.logger.info(f"Sentence Transformer {model_name} chargé avec succès")
#         except Exception as e:
#             self.logger.error(f"Erreur lors du chargement du Sentence Transformer: {e}")
#             self.sentence_transformer = None
    
#     def load_model(self, model_path: str = None):
#         """Charge le modèle avec priorité sur la recherche sémantique"""
#         path = model_path or self.config.MODEL_PATH
        
#         try:
#             # Vérifier si le modèle personnalisé existe
#             if os.path.exists(path) and os.path.exists(os.path.join(path, 'config.json')):
#                 self.logger.info(f"Chargement du modèle personnalisé depuis {path}")
                
#                 self.tokenizer = AutoTokenizer.from_pretrained(path)
#                 if self.tokenizer.pad_token is None:
#                     self.tokenizer.pad_token = self.tokenizer.eos_token
                
#                 self.model = AutoModelForCausalLM.from_pretrained(
#                     path,
#                     torch_dtype=torch.float32,
#                     low_cpu_mem_usage=True
#                 )
                
#                 self.is_trained = True
#                 self.logger.info("Modèle personnalisé chargé avec succès")
                
#             else:
#                 self.logger.warning(f"Modèle personnalisé non trouvé à {path}")
#                 self.logger.info("Mode recherche sémantique uniquement activé")
#                 self.is_trained = False
                
#         except Exception as e:
#             self.logger.error(f"Erreur lors du chargement du modèle: {e}")
#             self.logger.info("Utilisation de la recherche sémantique uniquement")
#             self.is_trained = False
    
#     def preprocess_query(self, query: str) -> str:
#         """Préprocesse et normalise la requête utilisateur"""
#         if not query:
#             return ""
        
#         query = query.strip()
        
#         # Normalisations spécifiques
#         replacements = {
#             r'\bmaroc\b': 'Ensemble du territoire national',
#             r'\broyaume du maroc\b': 'Ensemble du territoire national',
#             r'\bterritoire national\b': 'Ensemble du territoire national',
#             r'\bnational\b': 'Ensemble du territoire national',
#             r'\bcombien d\'habitants\b': 'population légale',
#             r'\bcombien de personnes\b': 'population légale',
#             r'\bpopulation totale\b': 'population légale',
#             r'\bnombre d\'habitants\b': 'population légale'
#         }
        
#         query_processed = query.lower()
#         for pattern, replacement in replacements.items():
#             query_processed = re.sub(pattern, replacement, query_processed, flags=re.IGNORECASE)
        
#         return query_processed
    
#     def extract_territory_from_query(self, query: str) -> Optional[str]:
#         """Extrait et normalise le territoire de la requête"""
#         query_lower = query.lower()
        
#         # Territoires du Maroc avec priorités
#         territory_patterns = [
#             (r'\bensemble du territoire national\b', 'Ensemble du territoire national'),
#             (r'\bterritoire national\b', 'Ensemble du territoire national'),
#             (r'\bmaroc\b', 'Ensemble du territoire national'),
#             (r'\broyaume du maroc\b', 'Ensemble du territoire national'),
#             (r'\bnational\b', 'Ensemble du territoire national'),
#         ]
        
#         for pattern, standard_name in territory_patterns:
#             if re.search(pattern, query_lower):
#                 return standard_name
        
#         # Chercher dans les territoires des données
#         if hasattr(self.data_processor, 'combined_data') and self.data_processor.combined_data:
#             territories = set([item.get('territoire', '') for item in self.data_processor.combined_data[:1000]])
            
#             for territory in territories:
#                 if territory and territory.lower() in query_lower:
#                     return territory
        
#         return None
    
#     def extract_indicator_from_query(self, query: str) -> Optional[str]:
#         """Extrait l'indicateur démographique de la requête"""
#         query_lower = query.lower()
        
#         # Patterns d'indicateurs spécifiques
#         indicator_patterns = {
#             'population_legale': [r'population légale', r'population legale', r'pop légale'],
#             'population_municipale': [r'population municipale', r'pop municipale'],
#             'age_0-4 ans': [r'0-4 ans', r'0 à 4 ans', r'moins de 5 ans'],
#             'age_5-9 ans': [r'5-9 ans', r'5 à 9 ans'],
#             'age_10-14 ans': [r'10-14 ans', r'10 à 14 ans'],
#             'age_15-19 ans': [r'15-19 ans', r'15 à 19 ans'],
#             'age_20-24 ans': [r'20-24 ans', r'20 à 24 ans'],
#             'age_25-29 ans': [r'25-29 ans', r'25 à 29 ans'],
#             'age_30-34 ans': [r'30-34 ans', r'30 à 34 ans'],
#             'age_35-39 ans': [r'35-39 ans', r'35 à 39 ans'],
#         }
        
#         for indicator, patterns in indicator_patterns.items():
#             for pattern in patterns:
#                 if re.search(pattern, query_lower):
#                     return indicator
        
#         # Patterns généraux
#         if re.search(r'pourcentage|%|taux', query_lower):
#             if re.search(r'\d+-\d+\s*ans', query_lower):
#                 age_match = re.search(r'(\d+-\d+)\s*ans', query_lower)
#                 if age_match:
#                     return f"age_{age_match.group(1)} ans"
        
#         return None
    
#     def extract_source_from_query(self, query: str) -> Optional[str]:
#         """NOUVEAU: Extrait la source (population/ménages) de la requête"""
#         query_lower = query.lower()
        
#         # Patterns pour identifier la source
#         if any(term in query_lower for term in ['ménage', 'menage', 'ménages', 'menages', 'foyer', 'foyers']):
#             return 'menages'
#         elif any(term in query_lower for term in ['population', 'habitants', 'démographie', 'démographique']):
#             return 'population'
        
#         return None
    
#     def is_greeting(self, query: str) -> bool:
#         """Détection améliorée des salutations"""
#         greetings = [
#             r'\bbonjour\b', r'\bbonsoir\b', r'\bsalut\b', r'\bhello\b', r'\bhi\b',
#             r'\baide\b', r'\baider\b', r'\bmerci\b', r'\bau revoir\b',
#             r'^(salut|hello|hi|bonjour|bonsoir)'  # <- CORRIGÉ: Guillemets ajoutés
#         ]
        
#         query_lower = query.lower().strip()
#         return any(re.search(pattern, query_lower) for pattern in greetings)
    
#     def find_best_match_semantic(self, query: str) -> Optional[Dict]:
#         """Recherche sémantique améliorée avec scoring multiple incluant la source"""
#         if not self.sentence_transformer or not self.qa_pairs:
#             return None
        
#         try:
#             processed_query = self.preprocess_query(query)
#             territory = self.extract_territory_from_query(query)
#             indicator = self.extract_indicator_from_query(query)
#             source = self.extract_source_from_query(query)  # NOUVEAU
            
#             # Créer une requête enrichie
#             enriched_query_parts = [processed_query]
#             if territory:
#                 enriched_query_parts.append(f"territoire: {territory}")
#             if indicator:
#                 enriched_query_parts.append(f"indicateur: {indicator}")
#             if source:  # NOUVEAU: Ajouter la source à la requête enrichie
#                 enriched_query_parts.append(f"source: {source}")
            
#             enriched_query = " | ".join(enriched_query_parts)
#             query_embedding = self.sentence_transformer.encode([enriched_query])
            
#             best_score = 0
#             best_match = None
            
#             for qa_pair in self.qa_pairs:
#                 if 'embedding' not in qa_pair:
#                     continue
                
#                 # Similarité sémantique
#                 similarity = np.dot(query_embedding[0], qa_pair['embedding'])
#                 score = float(similarity)
                
#                 # Bonus pour correspondance exacte de territoire
#                 if territory and qa_pair.get('territoire', '').lower() == territory.lower():
#                     score += 0.3
                
#                 # Bonus pour correspondance d'indicateur
#                 if indicator and qa_pair.get('indicateur', '').lower() == indicator.lower():
#                     score += 0.2
                
#                 # NOUVEAU: Bonus pour correspondance de source
#                 if source and qa_pair.get('source', '').lower() == source.lower():
#                     score += 0.25
                
#                 # Bonus pour genre si pertinent
#                 if 'ensemble' in query.lower() and qa_pair.get('genre', '') == 'ensemble':
#                     score += 0.1
                
#                 if score > best_score:
#                     best_score = score
#                     best_match = qa_pair
            
#             # Seuil de confiance adaptatif
#             threshold = getattr(self.config, 'SIMILARITY_THRESHOLD', 0.75)
#             if best_score > threshold:
#                 self.logger.info(f"Correspondance trouvée (score: {best_score:.3f})")
#                 return best_match
#             else:
#                 self.logger.info(f"Aucune correspondance suffisante (meilleur score: {best_score:.3f})")
#                 return None
                
#         except Exception as e:
#             self.logger.error(f"Erreur dans la recherche sémantique: {e}")
#             return None
    
#     def find_best_match_textual(self, query: str) -> Optional[Dict]:
#         """Recherche textuelle de fallback améliorée avec support des sources"""
#         if not self.qa_pairs:
#             return None
        
#         processed_query = self.preprocess_query(query)
#         territory = self.extract_territory_from_query(query)
#         indicator = self.extract_indicator_from_query(query)
#         source = self.extract_source_from_query(query)  # NOUVEAU
        
#         best_score = 0
#         best_match = None
        
#         for qa_pair in self.qa_pairs:
#             question_text = qa_pair.get('question', '').lower()
            
#             # Score de similarité textuelle
#             score = SequenceMatcher(None, processed_query, question_text).ratio()
            
#             # Bonus pour correspondances exactes
#             if territory and qa_pair.get('territoire', '').lower() == territory.lower():
#                 score += 0.4
#             if indicator and qa_pair.get('indicateur', '').lower() == indicator.lower():
#                 score += 0.3
#             if source and qa_pair.get('source', '').lower() == source.lower():  # NOUVEAU
#                 score += 0.25
            
#             # Bonus pour mots-clés importants
#             important_keywords = ['population', 'pourcentage', 'légale', 'municipale', 'ménage', 'menages']
#             for keyword in important_keywords:
#                 if keyword in processed_query and keyword in question_text:
#                     score += 0.1
            
#             if score > best_score and score > 0.6:
#                 best_score = score
#                 best_match = qa_pair
        
#         return best_match if best_score > 0.6 else None
    
#     def search_by_filters(self, territory: str = None, question_type: str = None, indicateur: str = None, source: str = None) -> Optional[Dict]:
#         """Recherche par filtres exacts incluant la source"""
#         if not self.qa_pairs:
#             return None
        
#         candidates = []
        
#         for qa_pair in self.qa_pairs:
#             match = True
            
#             # Filtrer par territoire
#             if territory:
#                 qa_territory = qa_pair.get('territoire', qa_pair.get('territory', ''))
#                 if qa_territory.lower() != territory.lower():
#                     match = False
            
#             # Filtrer par indicateur
#             if indicateur:
#                 qa_indicator = qa_pair.get('indicateur', qa_pair.get('variable', ''))
#                 if qa_indicator.lower() != indicateur.lower():
#                     match = False
            
#             # Filtrer par type de question
#             if question_type:
#                 if qa_pair.get('question_type', '') != question_type:
#                     match = False
            
#             # NOUVEAU: Filtrer par source
#             if source:
#                 qa_source = qa_pair.get('source', '')
#                 if qa_source.lower() != source.lower():
#                     match = False
            
#             if match:
#                 candidates.append(qa_pair)
        
#         # Retourner le premier candidat ou celui avec le meilleur score
#         if candidates:
#             # Prioriser 'ensemble' comme genre par défaut
#             ensemble_candidates = [c for c in candidates if c.get('genre', '') == 'ensemble']
#             return ensemble_candidates[0] if ensemble_candidates else candidates[0]
        
#         return None
    
#     def generate_smart_response(self, query: str) -> str:
#         """Génération de réponse avec stratégie intelligente incluant les sources"""
#         if self.is_greeting(query):
#             return random.choice(self.greeting_responses)
        
#         # 1. Recherche sémantique (priorité)
#         if self.sentence_transformer:
#             semantic_match = self.find_best_match_semantic(query)
#             if semantic_match:
#                 return semantic_match['answer']
        
#         # 2. Recherche par filtres exacts
#         territory = self.extract_territory_from_query(query)
#         indicator = self.extract_indicator_from_query(query)
#         source = self.extract_source_from_query(query)  # NOUVEAU
        
#         if territory or indicator or source:
#             filtered_result = self.search_by_filters(territory, None, indicator, source)
#             if filtered_result:
#                 return filtered_result['answer']
        
#         # 3. Recherche textuelle de fallback
#         textual_match = self.find_best_match_textual(query)
#         if textual_match:
#             return textual_match['answer']
        
#         # 4. Réponse guidée si aucune correspondance
#         guidance = self.generate_guidance_response(query, territory, indicator, source)
#         return guidance
    
#     def generate_guidance_response(self, query: str, territory: str = None, indicator: str = None, source: str = None) -> str:
#         """Génère une réponse d'aide contextuelle incluant les sources"""
#         guidance_parts = []
        
#         if territory:
#             guidance_parts.append(f"Territoire détecté: {territory}")
#         if indicator:
#             guidance_parts.append(f"Indicateur détecté: {indicator}")
#         if source:  # NOUVEAU
#             guidance_parts.append(f"Source détectée: {source}")
        
#         # Suggestions contextuelles
#         suggestions = []
        
#         if 'population' in query.lower():
#             suggestions.extend([
#                 "Quelle est la population légale de l'ensemble du territoire national ?",
#                 "Population municipale du Maroc ?"
#             ])
        
#         if 'ménage' in query.lower() or 'menage' in query.lower():
#             suggestions.extend([
#                 "Statistiques des ménages au niveau national ?",
#                 "Données sur les ménages du Maroc ?"
#             ])
        
#         if 'pourcentage' in query.lower() or '%' in query:
#             suggestions.extend([
#                 "Quel est le pourcentage de 0-4 ans au niveau national ?",
#                 "Pourcentage de la population de 15-19 ans ?"
#             ])
        
#         if not suggestions:
#             suggestions = [
#                 "Quelle est la population légale du Maroc ?",
#                 "Quel est le pourcentage de 20-24 ans au niveau national ?",
#                 "Population municipale de l'ensemble du territoire national ?",
#                 "Statistiques des ménages au Maroc ?"  # NOUVEAU
#             ]
        
#         response_parts = [
#             "Je n'ai pas trouvé de correspondance exacte pour votre question."
#         ]
        
#         if guidance_parts:
#             response_parts.append(" ".join(guidance_parts))
        
#         response_parts.extend([
#             "\n\nVoici des exemples de questions que je peux traiter :",
#             *[f"• {suggestion}" for suggestion in suggestions[:4]]  # Augmenté à 4 pour inclure les ménages
#         ])
        
#         return " ".join(response_parts)
    
#     def chat(self, query: str) -> str:
#         """Interface principale du chatbot - version corrigée avec support des sources"""
#         if not query or not query.strip():
#             return "Veuillez poser une question sur les statistiques démographiques ou les ménages du Maroc."
        
#         try:
#             # Log de la requête
#             self.logger.info(f"Question reçue: {query}")
            
#             # Générer la réponse avec la stratégie intelligente
#             response = self.generate_smart_response(query)
            
#             # Valider et nettoyer la réponse
#             cleaned_response = self.clean_response(response)
            
#             # Log de la réponse
#             self.logger.info(f"Réponse générée: {cleaned_response[:100]}...")
            
#             # Sauvegarder l'historique
#             self.save_conversation_history(query, cleaned_response)
            
#             return cleaned_response
            
#         except Exception as e:
#             self.logger.error(f"Erreur dans chat: {e}")
#             return "Je suis désolé, une erreur s'est produite. Veuillez reformuler votre question sur les données démographiques ou les ménages du Maroc."
    
#     def clean_response(self, response: str) -> str:
#         """Nettoie et valide la réponse"""
#         if not response:
#             return random.choice(self.default_responses)
        
#         # Supprimer les tokens spéciaux
#         cleaned = re.sub(r'<\|[^|]+\|>', '', response)
#         cleaned = re.sub(r'\[.*?\]', '', cleaned)
        
#         # Supprimer les répétitions
#         cleaned = re.sub(r'(.{10,}?)\1+', r'\1', cleaned)
        
#         # Nettoyer les espaces
#         cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
#         # Vérifier la longueur minimale
#         if len(cleaned) < 20:
#             return random.choice(self.default_responses)
        
#         return cleaned
    
#     def initialize_qa_pairs(self):
#         """Initialisation améliorée des paires Q&A avec support des sources"""
#         try:
#             self.logger.info("Initialisation des paires question-réponse...")
            
#             # Récupérer les données du processeur
#             generated_pairs = []
            
#             if hasattr(self.data_processor, 'combined_data') and self.data_processor.combined_data:
#                 self.logger.info(f"Traitement de {len(self.data_processor.combined_data)} éléments de données")
                
#                 for item in self.data_processor.combined_data:
#                     if item.get('question') and item.get('answer'):
#                         qa_pair = {
#                             'question': item.get('question', ''),
#                             'answer': item.get('answer', ''),
#                             'territoire': item.get('territoire', item.get('territory', 'Unknown')),
#                             'indicateur': item.get('indicateur', item.get('variable', 'unknown')),
#                             'genre': item.get('genre', item.get('sexe', 'ensemble')),
#                             'question_type': item.get('question_type', 'demographic'),
#                             'source': item.get('source', 'non spécifié')  # NOUVEAU: Inclure la source
#                         }
#                         generated_pairs.append(qa_pair)
            
#             # FAQ statiques avec sources
#             static_faqs = [
#                 {
#                     'question': "Qu'est-ce que le HCP ?",
#                     'answer': "Le Haut-Commissariat au Plan (HCP) est l'institution nationale marocaine chargée de la production et de la diffusion des statistiques officielles sur la population et les ménages.",
#                     'territoire': 'Maroc',
#                     'indicateur': 'information_hcp',
#                     'genre': 'ensemble',
#                     'question_type': 'hcp',
#                     'source': 'general'  # NOUVEAU: Source pour les FAQ
#                 },
#                 {
#                     'question': "Quelle est la différence entre population et ménages ?",
#                     'answer': "Les données de population concernent les individus (habitants, âges, genre), tandis que les données de ménages concernent les unités familiales et leurs caractéristiques (taille, composition, logement).",
#                     'territoire': 'Maroc',
#                     'indicateur': 'explication',
#                     'genre': 'ensemble',
#                     'question_type': 'definition',
#                     'source': 'general'  # NOUVEAU
#                 }
#             ]
            
#             # Fusionner les données
#             self.qa_pairs = static_faqs + generated_pairs
            
#             self.logger.info(f"{len(self.qa_pairs)} paires Q&A chargées")
            
#             # Afficher la répartition par source
#             sources_count = {}
#             for pair in self.qa_pairs:
#                 source = pair.get('source', 'non spécifié')
#                 sources_count[source] = sources_count.get(source, 0) + 1
            
#             if sources_count:
#                 self.logger.info("Répartition par source:")
#                 for source, count in sorted(sources_count.items(), key=lambda x: x[1], reverse=True):
#                     percentage = (count / len(self.qa_pairs)) * 100
#                     self.logger.info(f"   - {source}: {count} ({percentage:.1f}%)")
            
#             # Charger le sentence transformer et créer les embeddings
#             self.load_sentence_transformer()
            
#             if self.sentence_transformer and self.qa_pairs:
#                 self.logger.info("Création des embeddings sémantiques...")
                
#                 try:
#                     # Créer des textes enrichis pour les embeddings incluant la source
#                     enriched_texts = []
#                     for pair in self.qa_pairs:
#                         text_parts = [pair['question']]
                        
#                         territoire = pair.get('territoire', '')
#                         indicateur = pair.get('indicateur', '')
#                         source = pair.get('source', '')  # NOUVEAU
                        
#                         if territoire and territoire != 'Unknown':
#                             text_parts.append(f"territoire: {territoire}")
#                         if indicateur and indicateur != 'unknown':
#                             text_parts.append(f"indicateur: {indicateur}")
#                         if source and source != 'non spécifié':  # NOUVEAU
#                             text_parts.append(f"source: {source}")
                        
#                         enriched_text = " | ".join(text_parts)
#                         enriched_texts.append(enriched_text)
                    
#                     # Générer les embeddings
#                     embeddings = self.sentence_transformer.encode(enriched_texts, show_progress_bar=True)
                    
#                     # Assigner les embeddings aux paires
#                     for i, embedding in enumerate(embeddings):
#                         self.qa_pairs[i]['embedding'] = embedding
                    
#                     self.logger.info(f"{len(embeddings)} embeddings créés avec succès")
                    
#                 except Exception as e:
#                     self.logger.error(f"Erreur lors de la création des embeddings: {e}")
            
#             # Statistiques finales
#             territories = set(pair.get('territoire', 'Unknown') for pair in self.qa_pairs)
#             indicators = set(pair.get('indicateur', 'unknown') for pair in self.qa_pairs)
#             sources = set(pair.get('source', 'non spécifié') for pair in self.qa_pairs)  # NOUVEAU
            
#             self.logger.info(f"Couverture:")
#             self.logger.info(f"   • Territoires: {len(territories)}")
#             self.logger.info(f"   • Indicateurs: {len(indicators)}")
#             self.logger.info(f"   • Sources: {len(sources)}")  # NOUVEAU
            
#         except Exception as e:
#             self.logger.error(f"Erreur lors de l'initialisation des paires Q&A: {e}")
#             self.qa_pairs = []
    
#     def save_conversation_history(self, query: str, response: str):
#         """Sauvegarde l'historique avec métadonnées incluant les sources"""
#         if not getattr(self.config, 'SAVE_CONVERSATION_HISTORY', False):
#             return
            
#         try:
#             history_path = getattr(self.config, 'CONVERSATION_HISTORY_PATH', 'data/conversation_history.json')
            
#             # Charger l'historique existant
#             history = []
#             if os.path.exists(history_path):
#                 with open(history_path, 'r', encoding='utf-8') as f:
#                     history = json.load(f)
            
#             # Ajouter la nouvelle conversation avec métadonnées
#             conversation_entry = {
#                 'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
#                 'query': query,
#                 'response': response,
#                 'method': 'semantic_search' if self.sentence_transformer else 'textual_search',
#                 'territory_detected': self.extract_territory_from_query(query),
#                 'indicator_detected': self.extract_indicator_from_query(query),
#                 'source_detected': self.extract_source_from_query(query)  # NOUVEAU
#             }
            
#             history.append(conversation_entry)
            
#             # Garder seulement les 1000 dernières
#             if len(history) > 1000:
#                 history = history[-1000:]
            
#             # Créer le dossier si nécessaire
#             os.makedirs(os.path.dirname(history_path), exist_ok=True)
            
#             # Sauvegarder
#             with open(history_path, 'w', encoding='utf-8') as f:
#                 json.dump(history, f, indent=2, ensure_ascii=False)
                
#         except Exception as e:
#             self.logger.error(f"Erreur lors de la sauvegarde: {e}")
    
#     def get_statistics(self) -> Dict:
#         """Statistiques complètes du chatbot incluant les sources"""
#         stats = {
#             'model_loaded': self.model is not None,
#             'is_trained': self.is_trained,
#             'qa_pairs_count': len(self.qa_pairs),
#             'has_sentence_transformer': self.sentence_transformer is not None,
#             'embedding_count': sum(1 for pair in self.qa_pairs if 'embedding' in pair),
#             'base_model': getattr(self.config, 'BASE_MODEL', 'Unknown'),
#             'data_structure': 'nouvelle_structure_qa_pairs_optimisée_avec_sources'
#         }
        
#         if self.qa_pairs:
#             territories = set(pair.get('territoire', 'Unknown') for pair in self.qa_pairs)
#             indicators = set(pair.get('indicateur', 'unknown') for pair in self.qa_pairs)
#             genres = set(pair.get('genre', 'ensemble') for pair in self.qa_pairs)
#             sources = set(pair.get('source', 'non spécifié') for pair in self.qa_pairs)  # NOUVEAU
            
#             stats.update({
#                 'unique_territories': len(territories),
#                 'unique_indicators': len(indicators),
#                 'unique_genres': len(genres),
#                 'unique_sources': len(sources),  # NOUVEAU
#                 'coverage': {
#                     'territories_sample': sorted(list(territories))[:10],
#                     'indicators_sample': sorted(list(indicators))[:10],
#                     'genres': sorted(list(genres)),
#                     'sources': sorted(list(sources))  # NOUVEAU
#                 }
#             })
        
#         return stats
    
#     def get_help_message(self) -> str:
#         """Message d'aide détaillé incluant les sources"""
#         help_msg = """Assistant HCP - Guide d'utilisation

# Je suis spécialisé dans les statistiques démographiques et des ménages du Maroc du Haut-Commissariat au Plan.

# Types de questions supportées:
# • Population légale et municipale
# • Répartition par tranches d'âge
# • Pourcentages démographiques
# • Statistiques des ménages
# • Informations sur le HCP

# Exemples de questions efficaces:
# • "Quelle est la population légale du Maroc ?"
# • "Population municipale de l'ensemble du territoire national ?"
# • "Quel est le pourcentage de 0-4 ans au niveau national ?"
# • "Statistiques des ménages au Maroc ?"
# • "Données sur les ménages du territoire national ?"

# Conseils pour de meilleures réponses:
# • Mentionnez explicitement le territoire (ex: "Maroc", "ensemble du territoire national")
# • Soyez précis sur l'indicateur recherché (population légale/municipale, tranche d'âge)
# • Spécifiez si vous cherchez des données de population ou de ménages
# • Utilisez des questions complètes plutôt que des mots-clés

# Recherche intelligente:
# Le système utilise la recherche sémantique pour comprendre vos questions même si elles ne correspondent pas exactement aux données. Il peut identifier automatiquement:
# - Le territoire concerné
# - L'indicateur recherché  
# - La source de données (population ou ménages)"""
        
#         return help_msg


# def create_chatbot_with_config(config, data_processor):
#     """Crée et initialise le chatbot avec support des sources"""
#     chatbot = HCPChatbot(config, data_processor)
#     chatbot.load_model()
#     chatbot.initialize_qa_pairs()
#     return chatbot







# """
# Adaptation combinée : chatbot.py + app.py
# Fichier unique contenant deux modules adaptés à la nouvelle structure de données (qa_pairs)
# - HCPChatbotAdapted (classe)
# - Flask app adapted (fonction initialize_chatbot + routes)

# Utilise les champs normalisés produits par le DataProcessor adapté :
# - 'territory' (normalisé), 'original_territory'
# - 'variable' (indicateur), 'sexe'
# - 'source_data'
# - 'question', 'answer'
# - 'question_hash', 'question_type', 'indicators'

# Remarques :
# - Les fonctions d'extraction acceptent les deux variantes (fr/en) et tombent sur les champs normalisés.
# - Les embeddings sont normalisés L2 pour rendre la similarité cosinus stable.
# - Toute logique de fallback textuel/filtrage prend en compte 'source_data'.

# """

# # -----------------------------
# # chat/ chatbot adapted module
# # -----------------------------
# import os
# import re
# import json
# import time
# import random
# import logging
# from typing import Optional, Dict, List

# import numpy as np
# import torch
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from difflib import SequenceMatcher


# class HCPChatbotAdapted:
#     """Chatbot adapté à la structure 'qa_pairs' normalisée par HCPDataProcessor.

#     Principales différences :
#       - utilise fields normalisés ('territory','variable','sexe','source_data')
#       - embeddings normalisés L2 et comparaison cosinus
#       - recherche sémantique + fallback textuel + recherche par filtres
#       - sauvegarde historique standardisée
#     """

#     def __init__(self, config, data_processor):
#         self.config = config
#         self.data_processor = data_processor
#         self.tokenizer = None
#         self.model = None
#         self.qa_pairs: List[Dict] = []
#         self.is_trained = False
#         self.sentence_transformer: Optional[SentenceTransformer] = None

#         logging.basicConfig(level=getattr(config, 'LOG_LEVEL', logging.INFO))
#         self.logger = logging.getLogger('HCPChatbotAdapted')

#         self.default_responses = getattr(config, 'DEFAULT_RESPONSES', [
#             "Je ne trouve pas d'informations précises sur cette question dans ma base de données HCP."
#         ])

#         self.greeting_responses = getattr(config, 'GREETING_RESPONSES', [
#             "Bonjour ! Je suis l'assistant statistique du HCP."
#         ])

#     # ----------------------
#     # Utils embedding
#     # ----------------------
#     @staticmethod
#     def _l2_normalize(v: np.ndarray) -> np.ndarray:
#         v = np.asarray(v, dtype=float)
#         norm = np.linalg.norm(v)
#         if norm == 0:
#             return v
#         return v / norm

#     # ----------------------
#     # Loading models
#     # ----------------------
#     def load_sentence_transformer(self):
#         model_name = getattr(self.config, 'EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
#         try:
#             self.sentence_transformer = SentenceTransformer(model_name)
#             self.logger.info(f"SentenceTransformer '{model_name}' chargé")
#         except Exception as e:
#             self.logger.error(f"Erreur chargement SentenceTransformer: {e}")
#             self.sentence_transformer = None

#     def load_model(self, model_path: Optional[str] = None):
#         path = model_path or getattr(self.config, 'MODEL_PATH', None)
#         try:
#             if path and os.path.exists(path) and os.path.exists(os.path.join(path, 'config.json')):
#                 self.logger.info(f"Chargement du modèle depuis {path}")
#                 self.tokenizer = AutoTokenizer.from_pretrained(path)
#                 if self.tokenizer.pad_token is None:
#                     self.tokenizer.pad_token = self.tokenizer.eos_token

#                 self.model = AutoModelForCausalLM.from_pretrained(path, low_cpu_mem_usage=True)
#                 if torch.cuda.is_available():
#                     try:
#                         self.model.to('cuda')
#                     except Exception:
#                         pass

#                 self.is_trained = True
#                 self.logger.info("Modèle personnalisé chargé")
#             else:
#                 self.logger.info("Aucun modèle entraîné trouvé, mode recherche sémantique")
#                 self.is_trained = False
#         except Exception as e:
#             self.logger.error(f"Erreur lors du chargement du modèle: {e}")
#             self.is_trained = False

#     # ----------------------
#     # Query preprocessing & extraction
#     # ----------------------
#     def preprocess_query(self, query: str) -> str:
#         if not query:
#             return ''
#         q = query.strip()
#         # normalize common expressions
#         q = re.sub(r"\bmaroc\b|\broyaume du maroc\b|\bterritoire national\b|\bnational\b",
#                    'Ensemble du territoire national', q, flags=re.IGNORECASE)
#         return q

#     def extract_territory_from_query(self, query: str) -> Optional[str]:
#         q = query.lower()
#         # check common patterns
#         patterns = ['ensemble du territoire national', 'territoire national', 'maroc', 'royaume du maroc', 'national']
#         for p in patterns:
#             if p in q:
#                 return 'Ensemble du territoire national'

#         # look through processor territories (normalized values)
#         if hasattr(self.data_processor, 'combined_data') and self.data_processor.combined_data:
#             # collect unique territory values (first N for speed)
#             sample = self.data_processor.combined_data[:2000]
#             for item in sample:
#                 t = item.get('territory') or item.get('original_territory') or item.get('territoire')
#                 if t and t.lower() in q:
#                     return t
#         return None

#     def extract_indicator_from_query(self, query: str) -> Optional[str]:
#         q = query.lower()
#         # try mapping from config mapping keys/labels
#         mapping = getattr(self.config, 'HCP_INDICATOR_MAPPING', {})
#         # match by label or key
#         for key, label in mapping.items():
#             if label and label.lower() in q:
#                 return key
#             if key.replace('_', ' ') in q:
#                 return key

#         # age ranges
#         age_match = re.search(r'(\d{1,2})\s*[–-—-]?\s*(\d{1,2})\s*ans', q)
#         if age_match:
#             return f"age_{age_match.group(1)}_{age_match.group(2)}"

#         # keywords
#         if 'population légale' in q or 'population legale' in q or 'population totale' in q:
#             return 'population_legale'
#         if 'population municipale' in q:
#             return 'population_municipale'

#         return None

#     def extract_source_from_query(self, query: str) -> Optional[str]:
#         q = query.lower()
#         if any(x in q for x in ['ménage', 'menage', 'ménages', 'menages', 'foyer']):
#             return 'menages'
#         if any(x in q for x in ['population', 'habitants', 'démographie', 'demographie']):
#             return 'population'
#         return None

#     def is_greeting(self, query: str) -> bool:
#         q = (query or '').lower().strip()
#         greetings = ['bonjour', 'bonsoir', 'salut', 'hello', 'hi']
#         return any(g in q for g in greetings)

#     # ----------------------
#     # QA initialization and embeddings
#     # ----------------------
#     def initialize_qa_pairs(self):
#         self.logger.info("Initialisation des paires Q&A (adapted)")
#         pairs = []

#         # Static FAQ
#         static_faqs = getattr(self.config, 'STATIC_FAQ', [])
#         for s in static_faqs:
#             pairs.append({
#                 'question': s.get('question'), 'answer': s.get('answer'),
#                 'territory': s.get('territory', 'Maroc'), 'variable': s.get('variable', 'info'),
#                 'sexe': s.get('sexe', 'ensemble'), 'source_data': s.get('source', 'general'),
#             })

#         # From data_processor
#         if hasattr(self.data_processor, 'combined_data') and self.data_processor.combined_data:
#             for item in self.data_processor.combined_data:
#                 q = item.get('question')
#                 a = item.get('answer')
#                 if not q or not a:
#                     continue
#                 pair = {
#                     'question': q,
#                     'answer': a,
#                     'territory': item.get('territory') or item.get('original_territory') or item.get('territoire', 'Unknown'),
#                     'variable': item.get('variable') or item.get('indicateur') or item.get('column_original', 'unknown'),
#                     'sexe': item.get('sexe') or item.get('genre', 'ensemble'),
#                     'question_type': item.get('question_type', 'demographic'),
#                     'source_data': item.get('source_data') or item.get('source', 'non spécifié')
#                 }
#                 pairs.append(pair)

#         self.qa_pairs = pairs
#         self.logger.info(f"{len(self.qa_pairs)} paires Q&A chargées")

#         # Load embedding model and create embeddings if possible
#         self.load_sentence_transformer()
#         if self.sentence_transformer and self.qa_pairs:
#             try:
#                 texts = []
#                 for p in self.qa_pairs:
#                     parts = [p.get('question', '')]
#                     if p.get('territory'):
#                         parts.append(f"territory: {p['territory']}")
#                     if p.get('variable'):
#                         parts.append(f"indicator: {p['variable']}")
#                     if p.get('source_data'):
#                         parts.append(f"source: {p['source_data']}")
#                     texts.append(' | '.join(parts))

#                 embeddings = self.sentence_transformer.encode(texts, show_progress_bar=False)
#                 # normalize and attach
#                 for i, emb in enumerate(embeddings):
#                     self.qa_pairs[i]['embedding'] = self._l2_normalize(np.asarray(emb))
#                 self.logger.info(f"{len(embeddings)} embeddings créés et normalisés")
#             except Exception as e:
#                 self.logger.error(f"Erreur création embeddings: {e}")

#     # ----------------------
#     # Searchers
#     # ----------------------
#     def find_best_match_semantic(self, query: str) -> Optional[Dict]:
#         if not self.sentence_transformer or not self.qa_pairs:
#             return None
#         try:
#             processed = self.preprocess_query(query)
#             territory = self.extract_territory_from_query(query)
#             indicator = self.extract_indicator_from_query(query)
#             source = self.extract_source_from_query(query)

#             enriched = [processed]
#             if territory:
#                 enriched.append(f"territory: {territory}")
#             if indicator:
#                 enriched.append(f"indicator: {indicator}")
#             if source:
#                 enriched.append(f"source: {source}")
#             enriched_query = ' | '.join(enriched)

#             q_emb = self.sentence_transformer.encode([enriched_query])[0]
#             q_emb = self._l2_normalize(np.asarray(q_emb))

#             best = None
#             best_score = -1.0
#             for pair in self.qa_pairs:
#                 emb = pair.get('embedding')
#                 if emb is None:
#                     continue
#                 score = float(np.dot(q_emb, emb))
#                 # bonuses
#                 if territory and pair.get('territory') and territory.lower() == pair.get('territory').lower():
#                     score += 0.25
#                 if indicator and pair.get('variable') and indicator.lower() == pair.get('variable').lower():
#                     score += 0.15
#                 if source and pair.get('source_data') and source.lower() == pair.get('source_data').lower():
#                     score += 0.2
#                 if score > best_score:
#                     best_score = score
#                     best = pair

#             threshold = getattr(self.config, 'SIMILARITY_THRESHOLD', 0.75)
#             if best and best_score >= threshold:
#                 self.logger.debug(f"Semantic best score {best_score}")
#                 return best
#             return None
#         except Exception as e:
#             self.logger.error(f"Erreur semantic search: {e}")
#             return None

#     def find_best_match_textual(self, query: str) -> Optional[Dict]:
#         if not self.qa_pairs:
#             return None
#         pq = self.preprocess_query(query).lower()
#         territory = self.extract_territory_from_query(query)
#         indicator = self.extract_indicator_from_query(query)
#         source = self.extract_source_from_query(query)

#         best = None
#         best_score = 0.0
#         for pair in self.qa_pairs:
#             qtext = (pair.get('question') or '').lower()
#             score = SequenceMatcher(None, pq, qtext).ratio()
#             if territory and pair.get('territory') and territory.lower() == pair.get('territory').lower():
#                 score += 0.35
#             if indicator and pair.get('variable') and indicator.lower() == pair.get('variable').lower():
#                 score += 0.25
#             if source and pair.get('source_data') and source.lower() == pair.get('source_data').lower():
#                 score += 0.2
#             # keywords
#             for k in ['population', 'pourcentage', 'ménage', 'menage', 'municipale', 'légale']:
#                 if k in pq and k in qtext:
#                     score += 0.05
#             if score > best_score:
#                 best_score = score
#                 best = pair
#         if best_score > 0.6:
#             return best
#         return None

#     def search_by_filters(self, territory: Optional[str] = None, question_type: Optional[str] = None,
#                           indicateur: Optional[str] = None, source: Optional[str] = None) -> Optional[Dict]:
#         if not self.qa_pairs:
#             return None
#         candidates = []
#         for pair in self.qa_pairs:
#             ok = True
#             if territory and (pair.get('territory') or '').lower() != territory.lower():
#                 ok = False
#             if indicateur and (pair.get('variable') or '').lower() != indicateur.lower():
#                 ok = False
#             if question_type and (pair.get('question_type') or '').lower() != question_type.lower():
#                 ok = False
#             if source and (pair.get('source_data') or '').lower() != source.lower():
#                 ok = False
#             if ok:
#                 candidates.append(pair)
#         if not candidates:
#             return None
#         # prefer ensemble
#         ensemble = [c for c in candidates if (c.get('sexe') or '') == 'ensemble']
#         return ensemble[0] if ensemble else candidates[0]

#     # ----------------------
#     # Responses
#     # ----------------------
#     def generate_smart_response(self, query: str) -> str:
#         if self.is_greeting(query):
#             return random.choice(self.greeting_responses)

#         # semantic
#         if self.sentence_transformer:
#             sem = self.find_best_match_semantic(query)
#             if sem:
#                 return sem.get('answer')

#         # filters
#         territory = self.extract_territory_from_query(query)
#         indicator = self.extract_indicator_from_query(query)
#         source = self.extract_source_from_query(query)
#         filt = self.search_by_filters(territory=territory, indicateur=indicator, source=source)
#         if filt:
#             return filt.get('answer')

#         # textual fallback
#         text = self.find_best_match_textual(query)
#         if text:
#             return text.get('answer')

#         # guidance
#         return self.generate_guidance_response(query, territory, indicator, source)

#     def generate_guidance_response(self, query: str, territory: Optional[str], indicator: Optional[str], source: Optional[str]) -> str:
#         parts = []
#         if territory:
#             parts.append(f"Territoire détecté: {territory}")
#         if indicator:
#             parts.append(f"Indicateur détecté: {indicator}")
#         if source:
#             parts.append(f"Source détectée: {source}")
#         suggestions = [
#             "Quelle est la population légale de l'ensemble du territoire national ?",
#             "Population municipale du Maroc ?",
#             "Quel est le pourcentage de 0-4 ans au niveau national ?",
#             "Statistiques des ménages au Maroc ?"
#         ]
#         resp = ["Je n'ai pas trouvé de correspondance exacte pour votre question."]
#         if parts:
#             resp.append(' '.join(parts))
#         resp.append("Voici quelques exemples que je peux traiter:")
#         resp.extend([f"• {s}" for s in suggestions[:4]])
#         return ' '.join(resp)

#     def clean_response(self, response: str) -> str:
#         if not response or len(response.strip()) < 5:
#             return random.choice(self.default_responses)
#         s = re.sub(r"<\|[^|]+\|>", '', response)
#         s = re.sub(r"\s+", ' ', s).strip()
#         return s

#     def save_conversation_history(self, query: str, response: str):
#         if not getattr(self.config, 'SAVE_CONVERSATION_HISTORY', False):
#             return
#         history_path = getattr(self.config, 'CONVERSATION_HISTORY_PATH', 'data/conversation_history.json')
#         try:
#             os.makedirs(os.path.dirname(history_path), exist_ok=True)
#             history = []
#             if os.path.exists(history_path):
#                 with open(history_path, 'r', encoding='utf-8') as f:
#                     history = json.load(f)
#             history.append({
#                 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
#                 'query': query,
#                 'response': response,
#                 'territory_detected': self.extract_territory_from_query(query),
#                 'indicator_detected': self.extract_indicator_from_query(query),
#                 'source_detected': self.extract_source_from_query(query)
#             })
#             # keep last 1000
#             history = history[-1000:]
#             with open(history_path, 'w', encoding='utf-8') as f:
#                 json.dump(history, f, ensure_ascii=False, indent=2)
#         except Exception as e:
#             self.logger.error(f"Erreur sauvegarde historique: {e}")

#     def chat(self, query: str) -> str:
#         if not query or not query.strip():
#             return "Veuillez poser une question sur les statistiques démographiques ou les ménages du Maroc."
#         try:
#             self.logger.info(f"Question: {query}")
#             resp = self.generate_smart_response(query)
#             cleaned = self.clean_response(resp)
#             self.save_conversation_history(query, cleaned)
#             return cleaned
#         except Exception as e:
#             self.logger.error(f"Erreur chat: {e}")
#             return random.choice(self.default_responses)

#     # ----------------------
#     # Utilities
#     # ----------------------
#     def get_statistics(self) -> Dict:
#         stats = {
#             'model_loaded': self.model is not None,
#             'is_trained': self.is_trained,
#             'qa_pairs_count': len(self.qa_pairs),
#             'has_sentence_transformer': self.sentence_transformer is not None,
#             'embedding_count': sum(1 for p in self.qa_pairs if 'embedding' in p),
#             'base_model': getattr(self.config, 'BASE_MODEL', 'Unknown'),
#             'data_structure': 'qa_pairs_normalized'
#         }
#         if self.qa_pairs:
#             territories = set(p.get('territory','Unknown') for p in self.qa_pairs)
#             indicators = set(p.get('variable','unknown') for p in self.qa_pairs)
#             sources = set(p.get('source_data','non spécifié') for p in self.qa_pairs)
#             stats.update({
#                 'unique_territories': len(territories),
#                 'unique_indicators': len(indicators),
#                 'unique_sources': len(sources)
#             })
#         return stats















"""
Adaptation combinée : chatbot.py optimisé
Version améliorée avec :
- Algorithme de recherche multi-niveaux performant
- Correction automatique d'orthographe
- Utilisation optimisée du modèle entraîné
- Cache d'embeddings pour améliorer les performances
- Recherche sémantique hybride avec scoring avancé
- Gestion robuste des fautes de frappe et variations linguistiques
"""

import os
import re
import json
import time
import random
import logging
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
import pickle
from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from difflib import SequenceMatcher, get_close_matches
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata


class HCPChatbotOptimized:
    """Chatbot HCP optimisé avec algorithmes de recherche avancés et correction d'orthographe."""

    def __init__(self, config, data_processor):
        self.config = config
        self.data_processor = data_processor
        self.tokenizer = None
        self.model = None
        self.qa_pairs: List[Dict] = []
        self.is_trained = False
        self.sentence_transformer: Optional[SentenceTransformer] = None
        
        # Nouveaux composants d'optimisation
        self.embeddings_cache = {}
        self.search_index = defaultdict(list)  # Index inversé pour recherche rapide
        self.vocabulary = set()  # Vocabulaire pour correction orthographique
        self.territory_aliases = {}  # Alias pour territoires
        self.indicator_keywords = {}  # Mots-clés pour indicateurs
        self.nlp_pipeline = None  # Pipeline pour analyse linguistique

        logging.basicConfig(level=getattr(config, 'LOG_LEVEL', logging.INFO))
        self.logger = logging.getLogger('HCPChatbotOptimized')

        # Réponses par défaut améliorées
        self.default_responses = getattr(config, 'DEFAULT_RESPONSES', [
            "Je ne trouve pas d'informations précises sur cette question dans ma base de données HCP.",
            "Pouvez-vous reformuler votre question ou être plus spécifique ?",
            "Cette information n'est pas disponible dans mes données actuelles."
        ])

        self.greeting_responses = getattr(config, 'GREETING_RESPONSES', [
            "Bonjour ! Je suis l'assistant statistique du HCP. Comment puis-je vous aider ?",
            "Salut ! Je peux vous renseigner sur les statistiques démographiques du Maroc.",
            "Bonjour ! Posez-moi vos questions sur la population et les ménages marocains."
        ])

        # Dictionnaire pour correction orthographique
        self.spell_corrections = {
            # Territoires
            'maroc': ['marrok', 'marroc', 'marok'],
            'casablanca': ['casablanka', 'cazablanca', 'casablaca'],
            'rabat': ['rabatt', 'rabt'],
            'national': ['nacionall', 'nasional'],
            
            # Indicateurs
            'population': ['populacion', 'poplation', 'popolation'],
            'ménage': ['menage', 'manage', 'ménagé'],
            'ménages': ['menages', 'manages', 'ménagés'],
            'municipale': ['municipalle', 'municiple', 'municipalé'],
            'légale': ['legale', 'légalle', 'legalé'],
            'pourcentage': ['pourcantage', 'pourcentagé', 'porcantage'],
            'statistique': ['statistiqué', 'statistic', 'statistik'],
            'démographie': ['demographie', 'démografié', 'demografié']
        }

    # ----------------------
    # Optimisations de base
    # ----------------------
    
    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        """Normalisation L2 optimisée avec gestion des cas limites."""
        v = np.asarray(v, dtype=np.float32)  # Utiliser float32 pour moins de mémoire
        norm = np.linalg.norm(v)
        return v / (norm + 1e-8)  # Éviter division par zéro

    @lru_cache(maxsize=1000)
    def _normalize_text(self, text: str) -> str:
        """Normalisation de texte avec cache pour améliorer les performances."""
        if not text:
            return ''
        # Normalisation Unicode
        text = unicodedata.normalize('NFD', text.lower())
        # Suppression des accents
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        # Nettoyage
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _build_search_index(self):
        """Construction d'un index inversé pour accélérer la recherche."""
        self.logger.info("Construction de l'index de recherche...")
        self.search_index.clear()
        self.vocabulary.clear()
        
        for i, pair in enumerate(self.qa_pairs):
            # Indexation par mots-clés
            question = self._normalize_text(pair.get('question', ''))
            territory = self._normalize_text(pair.get('territory', ''))
            variable = self._normalize_text(pair.get('variable', ''))
            
            # Ajouter au vocabulaire
            words = question.split() + territory.split() + variable.split()
            self.vocabulary.update(words)
            
            # Index inversé
            for word in words:
                if len(word) > 2:  # Ignorer les mots très courts
                    self.search_index[word].append(i)
            
            # Index par territoire exact
            if territory:
                self.search_index[f"territory:{territory}"].append(i)
            
            # Index par variable exacte
            if variable:
                self.search_index[f"variable:{variable}"].append(i)

    def _spell_correct_word(self, word: str) -> str:
        """Correction orthographique d'un mot."""
        normalized_word = self._normalize_text(word)
        
        # Vérifier si le mot existe déjà
        if normalized_word in self.vocabulary:
            return word
        
        # Chercher dans le dictionnaire de corrections
        for correct, variants in self.spell_corrections.items():
            if normalized_word in [self._normalize_text(v) for v in variants]:
                return correct
        
        # Recherche par similarité dans le vocabulaire
        if self.vocabulary:
            matches = get_close_matches(normalized_word, self.vocabulary, n=1, cutoff=0.8)
            if matches:
                return matches[0]
        
        return word

    def _spell_correct_query(self, query: str) -> str:
        """Correction orthographique de la requête complète."""
        words = query.split()
        corrected_words = []
        
        for word in words:
            corrected = self._spell_correct_word(word)
            corrected_words.append(corrected)
        
        corrected_query = ' '.join(corrected_words)
        
        # Log si correction effectuée
        if corrected_query != query:
            self.logger.info(f"Correction orthographique: '{query}' -> '{corrected_query}'")
        
        return corrected_query

    # ----------------------
    # Chargement des modèles optimisé
    # ----------------------
    
    def load_sentence_transformer(self):
        """Chargement optimisé du modèle d'embedding."""
        model_name = getattr(self.config, 'EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        try:
            self.sentence_transformer = SentenceTransformer(model_name)
            
            # Optimiser pour l'inférence
            self.sentence_transformer.eval()
            if torch.cuda.is_available():
                try:
                    self.sentence_transformer = self.sentence_transformer.cuda()
                except Exception as e:
                    self.logger.warning(f"CUDA non disponible pour SentenceTransformer: {e}")
            
            self.logger.info(f"SentenceTransformer '{model_name}' chargé avec optimisations")
        except Exception as e:
            self.logger.error(f"Erreur chargement SentenceTransformer: {e}")
            self.sentence_transformer = None

    def load_model(self, model_path: Optional[str] = None):
        """Chargement optimisé du modèle de génération."""
        path = model_path or getattr(self.config, 'MODEL_PATH', None)
        try:
            if path and os.path.exists(path) and os.path.exists(os.path.join(path, 'config.json')):
                self.logger.info(f"Chargement du modèle depuis {path}")
                
                # Chargement optimisé
                self.tokenizer = AutoTokenizer.from_pretrained(
                    path, 
                    use_fast=True,  # Utiliser tokenizer rapide si disponible
                    padding_side='left'
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Modèle avec optimisations mémoire
                self.model = AutoModelForCausalLM.from_pretrained(
                    path, 
                    torch_dtype=torch.float16,  # Utiliser half precision
                    low_cpu_mem_usage=True,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                # Mode évaluation pour l'inférence
                self.model.eval()
                
                # Créer pipeline pour génération optimisée
                if torch.cuda.is_available():
                    self.nlp_pipeline = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=0,
                        torch_dtype=torch.float16
                    )
                
                self.is_trained = True
                self.logger.info("Modèle personnalisé chargé avec optimisations")
            else:
                self.logger.info("Aucun modèle entraîné trouvé, mode recherche sémantique uniquement")
                self.is_trained = False
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du modèle: {e}")
            self.is_trained = False

    # ----------------------
    # Extraction d'entités améliorée
    # ----------------------
    
    def extract_territory_from_query(self, query: str) -> Optional[str]:
        """Extraction de territoire avec gestion d'alias et correction orthographique."""
        q = self._normalize_text(query)
        
        # Patterns améliorés avec aliases
        territory_patterns = {
            'Ensemble du territoire national': [
                'ensemble du territoire national', 'territoire national', 'maroc', 
                'royaume du maroc', 'national', 'pays', 'tout le maroc'
            ],
            'Grand Casablanca': ['casablanca', 'casa', 'grand casablanca'],
            'Rabat-Salé-Kénitra': ['rabat', 'sale', 'kenitra', 'rabat sale', 'rabat sale kenitra'],
            'Fès-Meknès': ['fes', 'meknes', 'fes meknes'],
            'Marrakech-Safi': ['marrakech', 'safi', 'marrakech safi']
        }
        
        # Recherche avec correction orthographique
        corrected_query = self._spell_correct_query(q)
        
        for territory, aliases in territory_patterns.items():
            for alias in aliases:
                if alias in corrected_query:
                    return territory
        
        # Recherche dans les données avec similarité
        if hasattr(self.data_processor, 'combined_data') and self.data_processor.combined_data:
            territories = set()
            sample_size = min(1000, len(self.data_processor.combined_data))
            
            for item in self.data_processor.combined_data[:sample_size]:
                t = item.get('territory') or item.get('original_territory')
                if t:
                    territories.add(self._normalize_text(t))
            
            # Recherche par similarité
            matches = get_close_matches(corrected_query, territories, n=1, cutoff=0.7)
            if matches:
                # Retrouver le territoire original
                for item in self.data_processor.combined_data[:sample_size]:
                    t = item.get('territory') or item.get('original_territory')
                    if t and self._normalize_text(t) == matches[0]:
                        return t
        
        return None

    def extract_indicator_from_query(self, query: str) -> Optional[str]:
        """Extraction d'indicateur améliorée avec synonymes et correction orthographique."""
        q = self._normalize_text(query)
        corrected_query = self._spell_correct_query(q)
        
        # Mapping étendu des indicateurs avec synonymes
        indicator_patterns = {
            'population_legale': [
                'population legale', 'population légale', 'pop legale', 
                'habitants', 'nombre habitants', 'population totale'
            ],
            'population_municipale': [
                'population municipale', 'pop municipale', 'municipale'
            ],
            'menages': [
                'menage', 'ménage', 'menages', 'ménages', 'foyer', 'foyers',
                'famille', 'familles', 'nombre menages'
            ],
            'pourcentage_0_4': [
                '0 4 ans', '0-4 ans', '0 a 4 ans', 'zero quatre ans',
                'enfants 0 4', 'moins 5 ans'
            ],
            'pourcentage_5_9': [
                '5 9 ans', '5-9 ans', '5 a 9 ans', 'cinq neuf ans'
            ],
            'taux_urbanisation': [
                'urbanisation', 'taux urbanisation', 'urbain', 'ville'
            ]
        }
        
        # Recherche avec patterns
        for indicator, patterns in indicator_patterns.items():
            for pattern in patterns:
                if pattern in corrected_query:
                    return indicator
        
        # Recherche par regex pour tranches d'âge
        age_match = re.search(r'(\d{1,2})\s*[–\-—-]?\s*(\d{1,2})\s*ans?', corrected_query)
        if age_match:
            return f"pourcentage_{age_match.group(1)}_{age_match.group(2)}"
        
        # Recherche dans le mapping de configuration
        mapping = getattr(self.config, 'HCP_INDICATOR_MAPPING', {})
        for key, label in mapping.items():
            if label and self._normalize_text(label) in corrected_query:
                return key
            key_normalized = key.replace('_', ' ')
            if key_normalized in corrected_query:
                return key
        
        return None

    def extract_source_from_query(self, query: str) -> Optional[str]:
        """Extraction de source avec correction orthographique."""
        q = self._normalize_text(query)
        corrected_query = self._spell_correct_query(q)
        
        source_patterns = {
            'menages': [
                'menage', 'ménage', 'menages', 'ménages', 'foyer', 'foyers',
                'famille', 'familles', 'enquete menages', 'enquête ménages'
            ],
            'population': [
                'population', 'habitants', 'demographie', 'démographie',
                'recensement', 'rgph', 'statistiques population'
            ],
            'emploi': [
                'emploi', 'travail', 'chomage', 'chômage', 'activite', 'activité'
            ]
        }
        
        for source, patterns in source_patterns.items():
            for pattern in patterns:
                if pattern in corrected_query:
                    return source
        
        return None

    # ----------------------
    # Algorithme de recherche multi-niveaux
    # ----------------------
    
    def find_best_match_semantic_advanced(self, query: str) -> Optional[Dict]:
        """Recherche sémantique avancée avec scoring hybride."""
        if not self.sentence_transformer or not self.qa_pairs:
            return None
        
        try:
            # Préprocessing avec correction orthographique
            processed_query = self.preprocess_query(query)
            corrected_query = self._spell_correct_query(processed_query)
            
            # Extraction d'entités
            territory = self.extract_territory_from_query(corrected_query)
            indicator = self.extract_indicator_from_query(corrected_query)
            source = self.extract_source_from_query(corrected_query)
            
            # Construction de la requête enrichie
            enriched_parts = [corrected_query]
            if territory:
                enriched_parts.append(f"territoire: {territory}")
            if indicator:
                enriched_parts.append(f"indicateur: {indicator}")
            if source:
                enriched_parts.append(f"source: {source}")
            
            enriched_query = ' | '.join(enriched_parts)
            
            # Génération de l'embedding de la requête
            query_embedding = self.sentence_transformer.encode([enriched_query])[0]
            query_embedding = self._l2_normalize(query_embedding)
            
            # Scoring avancé avec multiple critères
            candidates = []
            for i, pair in enumerate(self.qa_pairs):
                if 'embedding' not in pair:
                    continue
                
                # Score sémantique de base
                semantic_score = float(np.dot(query_embedding, pair['embedding']))
                
                # Bonus contextuels
                context_bonus = 0.0
                
                # Bonus territoire (fort impact)
                if territory and pair.get('territory'):
                    if self._normalize_text(territory) == self._normalize_text(pair['territory']):
                        context_bonus += 0.3
                    elif territory.lower() in self._normalize_text(pair['territory']):
                        context_bonus += 0.15
                
                # Bonus indicateur (impact moyen)
                if indicator and pair.get('variable'):
                    if self._normalize_text(indicator) == self._normalize_text(pair['variable']):
                        context_bonus += 0.25
                    elif any(word in self._normalize_text(pair['variable']) 
                           for word in self._normalize_text(indicator).split()):
                        context_bonus += 0.1
                
                # Bonus source (impact moyen)
                if source and pair.get('source_data'):
                    if self._normalize_text(source) == self._normalize_text(pair['source_data']):
                        context_bonus += 0.2
                
                # Bonus mots-clés (impact faible)
                query_words = set(corrected_query.split())
                question_words = set(self._normalize_text(pair.get('question', '')).split())
                common_words = query_words.intersection(question_words)
                if common_words:
                    word_bonus = min(0.1, len(common_words) * 0.02)
                    context_bonus += word_bonus
                
                # Score final
                final_score = semantic_score + context_bonus
                
                candidates.append({
                    'pair': pair,
                    'score': final_score,
                    'semantic_score': semantic_score,
                    'context_bonus': context_bonus
                })
            
            # Trier par score et appliquer le seuil
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            if candidates:
                best = candidates[0]
                threshold = getattr(self.config, 'SIMILARITY_THRESHOLD', 0.7)
                
                self.logger.debug(
                    f"Meilleur match sémantique: score={best['score']:.3f} "
                    f"(sémantique={best['semantic_score']:.3f}, "
                    f"contexte={best['context_bonus']:.3f})"
                )
                
                if best['score'] >= threshold:
                    return best['pair']
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erreur recherche sémantique avancée: {e}")
            return None

    def find_best_match_index_search(self, query: str) -> Optional[Dict]:
        """Recherche rapide par index inversé."""
        if not self.search_index:
            return None
        
        try:
            processed_query = self.preprocess_query(query)
            corrected_query = self._spell_correct_query(processed_query)
            normalized_query = self._normalize_text(corrected_query)
            
            # Extraction d'entités pour recherche exacte
            territory = self.extract_territory_from_query(corrected_query)
            indicator = self.extract_indicator_from_query(corrected_query)
            
            # Score des candidats par index
            candidate_scores = defaultdict(float)
            
            # Recherche par territoire exact (priorité élevée)
            if territory:
                territory_key = f"territory:{self._normalize_text(territory)}"
                for idx in self.search_index.get(territory_key, []):
                    candidate_scores[idx] += 0.5
            
            # Recherche par indicateur exact (priorité élevée)
            if indicator:
                indicator_key = f"variable:{self._normalize_text(indicator)}"
                for idx in self.search_index.get(indicator_key, []):
                    candidate_scores[idx] += 0.4
            
            # Recherche par mots-clés
            query_words = normalized_query.split()
            for word in query_words:
                if len(word) > 2 and word in self.search_index:
                    for idx in self.search_index[word]:
                        candidate_scores[idx] += 0.1 / len(query_words)
            
            # Trouver le meilleur candidat
            if candidate_scores:
                best_idx = max(candidate_scores.keys(), key=lambda x: candidate_scores[x])
                best_score = candidate_scores[best_idx]
                
                if best_score >= 0.3:  # Seuil pour recherche par index
                    return self.qa_pairs[best_idx]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erreur recherche par index: {e}")
            return None

    def find_best_match_fuzzy_advanced(self, query: str) -> Optional[Dict]:
        """Recherche floue avancée avec multiple algorithmes."""
        if not self.qa_pairs:
            return None
        
        try:
            processed_query = self.preprocess_query(query)
            corrected_query = self._spell_correct_query(processed_query)
            normalized_query = self._normalize_text(corrected_query)
            
            # Extraction d'entités
            territory = self.extract_territory_from_query(corrected_query)
            indicator = self.extract_indicator_from_query(corrected_query)
            source = self.extract_source_from_query(corrected_query)
            
            best_pair = None
            best_score = 0.0
            
            for pair in self.qa_pairs:
                # Scores de similarité multiples
                question_text = self._normalize_text(pair.get('question', ''))
                
                # Similarité de séquence (Ratcliff-Obershelp)
                sequence_similarity = SequenceMatcher(None, normalized_query, question_text).ratio()
                
                # Similarité de mots (Jaccard)
                query_words = set(normalized_query.split())
                question_words = set(question_text.split())
                if query_words and question_words:
                    jaccard_similarity = len(query_words & question_words) / len(query_words | question_words)
                else:
                    jaccard_similarity = 0.0
                
                # Score de base (moyenne pondérée)
                base_score = 0.6 * sequence_similarity + 0.4 * jaccard_similarity
                
                # Bonus contextuels
                context_bonus = 0.0
                
                if territory and pair.get('territory'):
                    if self._normalize_text(territory) == self._normalize_text(pair['territory']):
                        context_bonus += 0.4
                    elif territory.lower() in self._normalize_text(pair['territory']):
                        context_bonus += 0.2
                
                if indicator and pair.get('variable'):
                    if self._normalize_text(indicator) == self._normalize_text(pair['variable']):
                        context_bonus += 0.3
                    elif any(word in self._normalize_text(pair['variable']) 
                           for word in self._normalize_text(indicator).split()):
                        context_bonus += 0.15
                
                if source and pair.get('source_data'):
                    if self._normalize_text(source) == self._normalize_text(pair['source_data']):
                        context_bonus += 0.2
                
                # Bonus mots-clés spéciaux
                special_keywords = ['population', 'pourcentage', 'menage', 'municipale', 'legale']
                for keyword in special_keywords:
                    if keyword in normalized_query and keyword in question_text:
                        context_bonus += 0.05
                
                # Score final
                final_score = base_score + context_bonus
                
                if final_score > best_score:
                    best_score = final_score
                    best_pair = pair
            
            # Seuil pour recherche floue
            threshold = getattr(self.config, 'FUZZY_THRESHOLD', 0.65)
            if best_score >= threshold:
                self.logger.debug(f"Meilleur match flou: score={best_score:.3f}")
                return best_pair
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erreur recherche floue avancée: {e}")
            return None

    def generate_with_trained_model(self, query: str, context: Optional[Dict] = None) -> Optional[str]:
        """Génération avec le modèle entraîné optimisée."""
        if not self.is_trained or not self.model:
            return None
        
        try:
            # Préparation du prompt avec contexte
            prompt_parts = ["Question:", query]
            
            if context:
                if context.get('territory'):
                    prompt_parts.append(f"Territoire: {context['territory']}")
                if context.get('variable'):
                    prompt_parts.append(f"Indicateur: {context['variable']}")
                if context.get('source_data'):
                    prompt_parts.append(f"Source: {context['source_data']}")
            
            prompt = "\n".join(prompt_parts) + "\nRéponse:"
            
            # Génération avec pipeline optimisé
            if self.nlp_pipeline:
                outputs = self.nlp_pipeline(
                    prompt,
                    max_length=len(prompt.split()) + 100,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False
                )
                
                if outputs and len(outputs) > 0:
                    generated_text = outputs[0]['generated_text']
                    return self.clean_response(generated_text)
            
            # Fallback avec tokenizer/model direct
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            return self.clean_response(response)
            
        except Exception as e:
            self.logger.error(f"Erreur génération modèle entraîné: {e}")
            return None

    # ----------------------
    # Pipeline de recherche principal
    # ----------------------
    
    def generate_smart_response(self, query: str) -> str:
        """Pipeline de recherche multi-niveaux optimisé."""
        if self.is_greeting(query):
            return random.choice(self.greeting_responses)
        
        # Niveau 1: Recherche sémantique avancée (la plus précise)
        if self.sentence_transformer:
            semantic_result = self.find_best_match_semantic_advanced(query)
            if semantic_result:
                # Tenter d'améliorer avec le modèle entraîné
                if self.is_trained:
                    enhanced_response = self.generate_with_trained_model(query, semantic_result)
                    if enhanced_response and len(enhanced_response.strip()) > 10:
                        return enhanced_response
                
                return semantic_result.get('answer')
        
        # Niveau 2: Recherche par index (rapide et efficace)
        index_result = self.find_best_match_index_search(query)
        if index_result:
            if self.is_trained:
                enhanced_response = self.generate_with_trained_model(query, index_result)
                if enhanced_response and len(enhanced_response.strip()) > 10:
                    return enhanced_response
            
            return index_result.get('answer')
        
        # Niveau 3: Recherche par filtres avec entités extraites
        territory = self.extract_territory_from_query(query)
        indicator = self.extract_indicator_from_query(query)
        source = self.extract_source_from_query(query)
        
        filter_result = self.search_by_filters(
            territory=territory, 
            indicateur=indicator, 
            source=source
        )
        if filter_result:
            if self.is_trained:
                enhanced_response = self.generate_with_trained_model(query, filter_result)
                if enhanced_response and len(enhanced_response.strip()) > 10:
                    return enhanced_response
            
            return filter_result.get('answer')
        
        # Niveau 4: Recherche floue avancée (fallback robuste)
        fuzzy_result = self.find_best_match_fuzzy_advanced(query)
        if fuzzy_result:
            if self.is_trained:
                enhanced_response = self.generate_with_trained_model(query, fuzzy_result)
                if enhanced_response and len(enhanced_response.strip()) > 10:
                    return enhanced_response
            
            return fuzzy_result.get('answer')
        
        # Niveau 5: Génération avec modèle entraîné seul (si disponible)
        if self.is_trained:
            model_response = self.generate_with_trained_model(query)
            if model_response and len(model_response.strip()) > 10:
                return model_response
        
        # Niveau 6: Réponse de guidage intelligente
        return self.generate_guidance_response(query, territory, indicator, source)

    # ----------------------
    # Fonctions utilitaires optimisées
    # ----------------------
    
    def search_by_filters(self, territory: Optional[str] = None, question_type: Optional[str] = None,
                          indicateur: Optional[str] = None, source: Optional[str] = None) -> Optional[Dict]:
        """Recherche par filtres avec scoring de pertinence."""
        if not self.qa_pairs:
            return None
        
        candidates = []
        for pair in self.qa_pairs:
            score = 0.0
            match = True
            
            # Filtrage strict avec scoring
            if territory:
                pair_territory = pair.get('territory', '')
                if self._normalize_text(territory) == self._normalize_text(pair_territory):
                    score += 0.4
                elif territory.lower() in pair_territory.lower():
                    score += 0.2
                else:
                    match = False
            
            if indicateur:
                pair_indicator = pair.get('variable', '')
                if self._normalize_text(indicateur) == self._normalize_text(pair_indicator):
                    score += 0.3
                elif any(word in self._normalize_text(pair_indicator) 
                        for word in self._normalize_text(indicateur).split()):
                    score += 0.15
                else:
                    match = False
            
            if question_type:
                pair_type = pair.get('question_type', '')
                if self._normalize_text(question_type) == self._normalize_text(pair_type):
                    score += 0.2
                else:
                    match = False
            
            if source:
                pair_source = pair.get('source_data', '')
                if self._normalize_text(source) == self._normalize_text(pair_source):
                    score += 0.25
                elif source.lower() in pair_source.lower():
                    score += 0.1
                else:
                    match = False
            
            if match:
                candidates.append((pair, score))
        
        if not candidates:
            return None
        
        # Trier par score et préférer 'ensemble' pour le sexe
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Préférer les résultats 'ensemble' à score égal
        best_score = candidates[0][1]
        best_candidates = [c for c in candidates if c[1] == best_score]
        
        ensemble_candidates = [c for c in best_candidates if c[0].get('sexe', '') == 'ensemble']
        
        if ensemble_candidates:
            return ensemble_candidates[0][0]
        
        return best_candidates[0][0]

    def preprocess_query(self, query: str) -> str:
        """Préprocessing de requête amélioré."""
        if not query:
            return ''
        
        # Normalisation de base
        q = query.strip()
        
        # Correction orthographique
        q = self._spell_correct_query(q)
        
        # Normalisation des expressions communes
        replacements = {
            r'\bmaroc\b|\broyaume du maroc\b|\bterritoire national\b|\bnational\b': 'Ensemble du territoire national',
            r'\bpop\b|\bpopul\b': 'population',
            r'\bmén\b|\bmenag\b': 'ménage',
            r'\bmunic\b': 'municipale',
            r'\blég\b|\blegal\b': 'légale',
            r'\bstats?\b': 'statistiques',
            r'\bdémog\b|\bdemog\b': 'démographie'
        }
        
        for pattern, replacement in replacements.items():
            q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)
        
        # Nettoyage final
        q = re.sub(r'\s+', ' ', q).strip()
        
        return q

    def generate_guidance_response(self, query: str, territory: Optional[str], 
                                 indicator: Optional[str], source: Optional[str]) -> str:
        """Génération de réponse de guidage intelligente."""
        parts = []
        
        # Information sur les entités détectées
        detected_entities = []
        if territory:
            detected_entities.append(f"Territoire: {territory}")
        if indicator:
            detected_entities.append(f"Indicateur: {indicator}")
        if source:
            detected_entities.append(f"Source: {source}")
        
        if detected_entities:
            parts.append("Éléments détectés dans votre question:")
            parts.extend([f"• {entity}" for entity in detected_entities])
            parts.append("")
        
        # Suggestions intelligentes basées sur les entités détectées
        suggestions = []
        
        if territory == "Ensemble du territoire national":
            suggestions.extend([
                "Quelle est la population légale de l'ensemble du territoire national ?",
                "Quel est le nombre de ménages au niveau national ?",
                "Population municipale du Maroc selon le RGPH ?"
            ])
        elif territory:
            suggestions.extend([
                f"Population légale de {territory} ?",
                f"Nombre de ménages dans {territory} ?",
                f"Statistiques démographiques de {territory} ?"
            ])
        
        if indicator:
            if "population" in indicator.lower():
                suggestions.extend([
                    "Population légale par région ?",
                    "Evolution de la population municipale ?",
                    "Répartition de la population par sexe ?"
                ])
            elif "menage" in indicator.lower():
                suggestions.extend([
                    "Taille moyenne des ménages ?",
                    "Nombre de ménages par région ?",
                    "Caractéristiques des ménages marocains ?"
                ])
        
        if not suggestions:
            suggestions = [
                "Quelle est la population légale de l'ensemble du territoire national ?",
                "Nombre de ménages au Maroc selon le RGPH ?",
                "Population municipale par région ?",
                "Statistiques démographiques du Maroc ?",
                "Répartition de la population par tranches d'âge ?"
            ]
        
        # Construction de la réponse
        base_responses = [
            "Je n'ai pas trouvé de correspondance exacte pour votre question.",
            "Cette information spécifique n'est pas disponible dans ma base de données.",
            "Pouvez-vous reformuler votre question de manière plus spécifique ?"
        ]
        
        parts.append(random.choice(base_responses))
        parts.append("")
        parts.append("Voici quelques exemples de questions que je peux traiter:")
        
        # Sélectionner les meilleures suggestions (max 5)
        selected_suggestions = suggestions[:5]
        parts.extend([f"• {s}" for s in selected_suggestions])
        
        # Conseils pour améliorer la question
        tips = []
        if not territory:
            tips.append("Spécifiez le territoire (région, province, ou 'Maroc' pour le niveau national)")
        if not indicator:
            tips.append("Précisez l'indicateur recherché (population, ménages, âge, etc.)")
        
        if tips:
            parts.append("")
            parts.append("Pour de meilleurs résultats, essayez de:")
            parts.extend([f"• {tip}" for tip in tips])
        
        return "\n".join(parts)

    def is_greeting(self, query: str) -> bool:
        """Détection de salutation améliorée."""
        if not query:
            return False
        
        q = self._normalize_text(query)
        
        # Patterns de salutation étendus
        greeting_patterns = [
            'bonjour', 'bonsoir', 'salut', 'hello', 'hi', 'hey',
            'bonne journee', 'bonne soiree', 'comment allez vous',
            'comment ca va', 'comment vous allez', 'ça va'
        ]
        
        # Questions courtes considérées comme salutations
        if len(q.split()) <= 3:
            return any(pattern in q for pattern in greeting_patterns)
        
        return False

    def clean_response(self, response: str) -> str:
        """Nettoyage de réponse amélioré."""
        if not response or len(response.strip()) < 5:
            return random.choice(self.default_responses)
        
        # Nettoyage des tokens spéciaux
        s = re.sub(r"<\|[^|]+\|>", '', response)
        s = re.sub(r"\[.*?\]", '', s)  # Supprimer les références entre crochets
        s = re.sub(r"\(.*?\)", '', s)  # Supprimer les parenthèses explicatives longues
        
        # Nettoyage des répétitions
        s = re.sub(r'\b(\w+)\s+\1\b', r'\1', s)  # Mots répétés
        s = re.sub(r'\.{2,}', '.', s)  # Points multiples
        
        # Nettoyage des espaces
        s = re.sub(r'\s+', ' ', s).strip()
        
        # Limitation de longueur si nécessaire
        max_length = getattr(self.config, 'MAX_RESPONSE_LENGTH', 500)
        if len(s) > max_length:
            sentences = s.split('.')
            truncated = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) + 1 <= max_length:
                    truncated.append(sentence)
                    current_length += len(sentence) + 1
                else:
                    break
            
            s = '. '.join(truncated)
            if s and not s.endswith('.'):
                s += '.'
        
        # Vérification finale
        if len(s.strip()) < 10:
            return random.choice(self.default_responses)
        
        return s

    # ----------------------
    # Initialisation optimisée
    # ----------------------
    
    def initialize_qa_pairs(self):
        """Initialisation optimisée des paires Q&A avec cache et indexation."""
        self.logger.info("Initialisation optimisée des paires Q&A")
        pairs = []

        # FAQ statique
        static_faqs = getattr(self.config, 'STATIC_FAQ', [])
        for s in static_faqs:
            pairs.append({
                'question': s.get('question'), 
                'answer': s.get('answer'),
                'territory': s.get('territory', 'Maroc'), 
                'variable': s.get('variable', 'info'),
                'sexe': s.get('sexe', 'ensemble'), 
                'source_data': s.get('source', 'general'),
                'question_type': 'faq'
            })

        # Données du processeur
        if hasattr(self.data_processor, 'combined_data') and self.data_processor.combined_data:
            self.logger.info(f"Traitement de {len(self.data_processor.combined_data)} éléments de données")
            
            for i, item in enumerate(self.data_processor.combined_data):
                q = item.get('question')
                a = item.get('answer')
                if not q or not a:
                    continue
                
                pair = {
                    'question': q,
                    'answer': a,
                    'territory': item.get('territory') or item.get('original_territory') or item.get('territoire', 'Unknown'),
                    'variable': item.get('variable') or item.get('indicateur') or item.get('column_original', 'unknown'),
                    'sexe': item.get('sexe') or item.get('genre', 'ensemble'),
                    'question_type': item.get('question_type', 'demographic'),
                    'source_data': item.get('source_data') or item.get('source', 'non spécifié')
                }
                pairs.append(pair)
                
                # Log du progrès pour gros datasets
                if (i + 1) % 1000 == 0:
                    self.logger.info(f"Traité {i + 1} éléments...")

        self.qa_pairs = pairs
        self.logger.info(f"{len(self.qa_pairs)} paires Q&A chargées")

        # Construction de l'index de recherche
        self._build_search_index()

        # Chargement et création des embeddings
        self.load_sentence_transformer()
        if self.sentence_transformer and self.qa_pairs:
            self._create_embeddings_with_cache()

    def _create_embeddings_with_cache(self):
        """Création d'embeddings avec système de cache pour améliorer les performances."""
        cache_path = getattr(self.config, 'EMBEDDINGS_CACHE_PATH', 'data/embeddings_cache.pkl')
        
        try:
            # Tentative de chargement du cache
            if os.path.exists(cache_path):
                self.logger.info("Chargement du cache d'embeddings...")
                with open(cache_path, 'rb') as f:
                    cached_embeddings = pickle.load(f)
                    
                # Vérification de la compatibilité du cache
                if len(cached_embeddings) == len(self.qa_pairs):
                    for i, emb in enumerate(cached_embeddings):
                        self.qa_pairs[i]['embedding'] = self._l2_normalize(np.asarray(emb))
                    
                    self.logger.info(f"Cache d'embeddings chargé avec succès ({len(cached_embeddings)} embeddings)")
                    return
                else:
                    self.logger.info("Cache d'embeddings incompatible, recalcul nécessaire")
            
            # Création des embeddings
            self.logger.info("Création des embeddings...")
            texts = []
            
            for pair in self.qa_pairs:
                # Construction du texte enrichi pour l'embedding
                parts = [pair.get('question', '')]
                
                territory = pair.get('territory')
                if territory and territory != 'Unknown':
                    parts.append(f"territoire: {territory}")
                
                variable = pair.get('variable')
                if variable and variable != 'unknown':
                    parts.append(f"indicateur: {variable}")
                
                source = pair.get('source_data')
                if source and source != 'non spécifié':
                    parts.append(f"source: {source}")
                
                texts.append(' | '.join(parts))
            
            # Création par batches pour optimiser la mémoire
            batch_size = getattr(self.config, 'EMBEDDING_BATCH_SIZE', 32)
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.sentence_transformer.encode(
                    batch_texts, 
                    show_progress_bar=False,
                    batch_size=len(batch_texts)
                )
                all_embeddings.extend(batch_embeddings)
                
                # Log du progrès
                if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(texts):
                    processed = min(i + batch_size, len(texts))
                    self.logger.info(f"Embeddings créés: {processed}/{len(texts)}")
            
            # Normalisation et attachement
            for i, emb in enumerate(all_embeddings):
                self.qa_pairs[i]['embedding'] = self._l2_normalize(np.asarray(emb, dtype=np.float32))
            
            # Sauvegarde du cache
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(all_embeddings, f)
                self.logger.info(f"Cache d'embeddings sauvegardé: {cache_path}")
            except Exception as e:
                self.logger.warning(f"Impossible de sauvegarder le cache: {e}")
            
            self.logger.info(f"{len(all_embeddings)} embeddings créés et normalisés avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la création des embeddings: {e}")

    # ----------------------
    # Interface principale optimisée
    # ----------------------
    
    def chat(self, query: str) -> str:
        """Interface de chat principale optimisée."""
        if not query or not query.strip():
            return "Veuillez poser une question sur les statistiques démographiques ou les ménages du Maroc."
        
        try:
            self.logger.info(f"Question reçue: {query}")
            start_time = time.time()
            
            # Génération de la réponse
            response = self.generate_smart_response(query)
            cleaned_response = self.clean_response(response)
            
            # Mesure des performances
            processing_time = time.time() - start_time
            self.logger.info(f"Réponse générée en {processing_time:.3f}s")
            
            # Sauvegarde de l'historique
            self.save_conversation_history(query, cleaned_response)
            
            return cleaned_response
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement de la question: {e}")
            return random.choice(self.default_responses)

    def save_conversation_history(self, query: str, response: str):
        """Sauvegarde d'historique optimisée avec métadonnées enrichies."""
        if not getattr(self.config, 'SAVE_CONVERSATION_HISTORY', False):
            return
        
        history_path = getattr(self.config, 'CONVERSATION_HISTORY_PATH', 'data/conversation_history.json')
        
        try:
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            
            # Chargement de l'historique existant
            history = []
            if os.path.exists(history_path):
                with open(history_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            # Nouvelle entrée avec métadonnées enrichies
            entry = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'query': query,
                'query_corrected': self._spell_correct_query(query),
                'response': response,
                'territory_detected': self.extract_territory_from_query(query),
                'indicator_detected': self.extract_indicator_from_query(query),
                'source_detected': self.extract_source_from_query(query),
                'response_length': len(response),
                'model_used': 'trained' if self.is_trained else 'search_only'
            }
            
            history.append(entry)
            
            # Limitation de l'historique (garder les 1000 derniers)
            max_history = getattr(self.config, 'MAX_HISTORY_SIZE', 1000)
            history = history[-max_history:]
            
            # Sauvegarde
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de l'historique: {e}")

    # ----------------------
    # Statistiques et monitoring
    # ----------------------
    
    def get_statistics(self) -> Dict:
        """Statistiques détaillées du chatbot."""
        stats = {
            'model_loaded': self.model is not None,
            'is_trained': self.is_trained,
            'qa_pairs_count': len(self.qa_pairs),
            'has_sentence_transformer': self.sentence_transformer is not None,
            'embedding_count': sum(1 for p in self.qa_pairs if 'embedding' in p),
            'vocabulary_size': len(self.vocabulary),
            'search_index_size': len(self.search_index),
            'base_model': getattr(self.config, 'BASE_MODEL', 'Unknown'),
            'embedding_model': getattr(self.config, 'EMBEDDING_MODEL', 'Unknown'),
            'data_structure': 'qa_pairs_normalized_optimized'
        }
        
        # Statistiques des données
        if self.qa_pairs:
            territories = set(p.get('territory', 'Unknown') for p in self.qa_pairs)
            indicators = set(p.get('variable', 'unknown') for p in self.qa_pairs)
            sources = set(p.get('source_data', 'non spécifié') for p in self.qa_pairs)
            question_types = set(p.get('question_type', 'unknown') for p in self.qa_pairs)
            
            stats.update({
                'unique_territories': len(territories),
                'unique_indicators': len(indicators),
                'unique_sources': len(sources),
                'unique_question_types': len(question_types),
                'territories': list(territories)[:10],  # Échantillon
                'indicators': list(indicators)[:10],
                'sources': list(sources)
            })
        
        # Statistiques des performances
        if hasattr(self, '_performance_stats'):
            stats.update(self._performance_stats)
        
        return stats

    def get_performance_report(self) -> Dict:
        """Rapport de performance détaillé."""
        history_path = getattr(self.config, 'CONVERSATION_HISTORY_PATH', 'data/conversation_history.json')
        
        report = {
            'total_queries': 0,
            'avg_response_length': 0,
            'model_usage': {'trained': 0, 'search_only': 0},
            'detected_entities': {
                'territories': defaultdict(int),
                'indicators': defaultdict(int),
                'sources': defaultdict(int)
            },
            'common_corrections': defaultdict(int)
        }
        
        try:
            if os.path.exists(history_path):
                with open(history_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                
                report['total_queries'] = len(history)
                
                if history:
                    # Analyse des réponses
                    response_lengths = [entry.get('response_length', 0) for entry in history]
                    report['avg_response_length'] = sum(response_lengths) / len(response_lengths)
                    
                    # Usage du modèle
                    for entry in history:
                        model = entry.get('model_used', 'search_only')
                        report['model_usage'][model] += 1
                    
                    # Entités détectées
                    for entry in history:
                        territory = entry.get('territory_detected')
                        if territory:
                            report['detected_entities']['territories'][territory] += 1
                        
                        indicator = entry.get('indicator_detected')
                        if indicator:
                            report['detected_entities']['indicators'][indicator] += 1
                        
                        source = entry.get('source_detected')
                        if source:
                            report['detected_entities']['sources'][source] += 1
                        
                        # Corrections orthographiques
                        original = entry.get('query', '')
                        corrected = entry.get('query_corrected', '')
                        if original != corrected:
                            report['common_corrections'][f"{original} -> {corrected}"] += 1
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du rapport: {e}")
        
        return report
















# import os
# import re
# import json
# import time
# import logging
# import pickle
# from typing import Optional, Dict, List, Tuple
# from collections import defaultdict
# from functools import lru_cache
# import unicodedata
# import random

# import numpy as np
# import torch
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from difflib import SequenceMatcher, get_close_matches


# class HCPChatbotOptimized:
#     """Chatbot HCP optimisé.
#     Les principales fonctions sont documentées au fur et à mesure.
#     """

#     def __init__(self, config, data_processor):
#         self.config = config
#         self.data_processor = data_processor
#         self.tokenizer = None
#         self.model = None
#         self.qa_pairs: List[Dict] = []
#         self.is_trained = False
#         self.sentence_transformer: Optional[SentenceTransformer] = None

#         # Index et caches
#         self.embeddings_cache = {}
#         self.search_index = defaultdict(list)
#         self.vocabulary = set()
#         self.territory_list = []  # liste normalisée des territoires disponibles

#         logging.basicConfig(level=getattr(config, 'LOG_LEVEL', logging.INFO))
#         self.logger = logging.getLogger('HCPChatbotOptimized')

#         # DEFAULTS
#         self.default_responses = getattr(config, 'DEFAULT_RESPONSES', [
#             "Je ne trouve pas d'informations précises sur cette question dans ma base de données HCP.",
#             "Pouvez-vous reformuler votre question ou être plus spécifique ?",
#             "Cette information n'est pas disponible dans mes données actuelles."
#         ])

#         self.greeting_responses = getattr(config, 'GREETING_RESPONSES', [
#             "Bonjour ! Je suis l'assistant statistique du HCP. Comment puis-je vous aider ?",
#             "Salut ! Je peux vous renseigner sur les statistiques démographiques du Maroc.",
#             "Bonjour ! Posez-moi vos questions sur la population et les ménages marocains."
#         ])

#         # Carte des corrections simples (peut être complétée dynamiquement)
#         self.spell_corrections = getattr(config, 'SPELL_CORRECTIONS', {})

#         # Construction d'un mapping initial d'indicateurs 
#         self.indicator_keywords = self._build_indicator_keywords_default()

#         # Paramètres
#         self.SIMILARITY_THRESHOLD = getattr(self.config, 'SIMILARITY_THRESHOLD', 0.70)
#         self.FUZZY_THRESHOLD = getattr(self.config, 'FUZZY_THRESHOLD', 0.65)

#     # ----------------------
#     # Normalisation et utilitaires
#     # ----------------------

#     @staticmethod
#     def _l2_normalize(v: np.ndarray) -> np.ndarray:
#         v = np.asarray(v, dtype=np.float32)
#         norm = np.linalg.norm(v)
#         return v / (norm + 1e-8)

#     @lru_cache(maxsize=4096)
#     def _normalize_text(self, text: str) -> str:
#         if not text:
#             return ''
#         t = text.strip().lower()
#         # Unicode normalization
#         t = unicodedata.normalize('NFD', t)
#         # Keep apostrophes though: convert curly quotes first
#         t = t.replace("’", "'").replace('`', "'")
#         # Remove diacritics but keep letters
#         t = ''.join(ch for ch in t if unicodedata.category(ch) != 'Mn')
#         # Replace punctuation except apostrophe and hyphen
#         t = re.sub(r"[^\w\s\'-]", ' ', t)
#         # Normalize whitespace
#         t = re.sub(r'\s+', ' ', t).strip()
#         return t

#     def _tokenize(self, text: str) -> List[str]:
#         t = self._normalize_text(text)
#         return [w for w in re.split(r"[\s-]+", t) if w]

#     # ----------------------
#     # Construction des mots-clés d'indicateurs
#     # ----------------------

#     def _build_indicator_keywords_default(self) -> Dict[str, List[str]]:
#         """Construit un mapping indicateur -> listes de mots-clés/synonymes
#         Basé sur les champs fournis par l'utilisateur (population + ménages).
#         """
#         # Liste illustrant une grande partie des champs donnés
#         mapping = {
#             'population_legale': ['population legale', 'population légale', 'population_légale', 'population totale', 'habitants'],
#             'population_municipale': ['population municipale', 'municipale', 'population_municipale'],
#             'population_par_sexe': ['sexe', 'masculin', 'feminin', 'hommes', 'femmes'],
#             'age_quinquennal': ['âge quinquennal', 'tranche d\'age', '0-4', '5-9', '10-14', '15-19'],
#             'taux_analphabetisme_10_plus': ['taux d\'analphabétisme', 'analphabetisme', 'analphabétisme'],
#             'taux_analphabetisme_15_plus': ['analphabétisme 15 ans', 'taux analphabétisme 15'],
#             'taux_scolarisation_6_11': ['scolarisation 6-11', 'taux de scolarisation 6 11', 'scolarisation'],
#             'population_15_plus': ['population 15 ans et plus', '15 ans et plus', '15+'],
#             'taux_chomage': ['taux de chomage', 'chomage', 'chômage', 'taux de chômage'],
#             'taille_moyenne_menages': ['taille moyenne', 'taille moyenne des menages', 'taille menage'],
#             'type_logement': ['type de logement', 'type logement', 'logement', 'villa', 'appartement', 'maison'],
#             'acces_eau': ['eau courante', 'eau potable', 'eau', 'eau courante'],
#             'electricite': ['electr', 'electricit', 'électricité', 'electricite'],
#             'mode_evacuation_dechets': ['dechets', 'ordures', 'evacuation des dechets'],
#             'combustible_cuisson': ['combustible', 'gaz', 'bois', 'charbon'],
#             'niveau_etudes': ['primai', 'second', 'superieur', 'niveau d\'etudes', 'prescolaire'],
#             'langues_lues_ecrites': ['langues lues', 'langues ecrites', 'langues lues et écrites'],
#             'handicap': ['prevalence du handicap', 'handicap', 'taux de prevalence du handicap'],
#             # Ajouter d'autres correspondances au besoin
#         }

#         # Ajouter variantes sans accents et courtes
#         expanded = {}
#         for k, vs in mapping.items():
#             s = set()
#             for v in vs:
#                 s.add(self._normalize_text(v))
#                 s.add(re.sub(r"\s+", ' ', re.sub(r"[^a-z0-9 ]", '', v.lower())))
#             expanded[k] = list(s)
#         return expanded

#     # ----------------------
#     # Chargement des modèles
#     # ----------------------

#     def load_sentence_transformer(self):
#         model_name = getattr(self.config, 'EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
#         try:
#             self.sentence_transformer = SentenceTransformer(model_name)
#             self.sentence_transformer.eval()
#             if torch.cuda.is_available():
#                 try:
#                     self.sentence_transformer = self.sentence_transformer.cuda()
#                 except Exception:
#                     self.logger.warning('CUDA indisponible pour SentenceTransformer')
#             self.logger.info(f"SentenceTransformer '{model_name}' chargé")
#         except Exception as e:
#             self.logger.error(f"Erreur chargement SentenceTransformer: {e}")
#             self.sentence_transformer = None

#     def load_model(self, model_path: Optional[str] = None):
#         path = model_path or getattr(self.config, 'MODEL_PATH', None)
#         try:
#             if path and os.path.exists(path) and os.path.exists(os.path.join(path, 'config.json')):
#                 self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
#                 if self.tokenizer.pad_token is None:
#                     self.tokenizer.pad_token = self.tokenizer.eos_token
#                 self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto' if torch.cuda.is_available() else None)
#                 self.model.eval()
#                 if torch.cuda.is_available():
#                     self.nlp_pipeline = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, device=0)
#                 self.is_trained = True
#                 self.logger.info('Modèle personnalisé chargé')
#             else:
#                 self.logger.info('Aucun modèle entrainé trouvé (mode recherche uniquement)')
#                 self.is_trained = False
#         except Exception as e:
#             self.logger.error(f'Erreur chargement modèle: {e}')
#             self.is_trained = False

#     # ----------------------
#     # Indexation
#     # ----------------------

#     def _build_search_index(self):
#         self.logger.info('Construction index inversé...')
#         self.search_index.clear()
#         self.vocabulary.clear()

#         for i, pair in enumerate(self.qa_pairs):
#             q = self._normalize_text(pair.get('question', ''))
#             t = self._normalize_text(pair.get('territory', '') or '')
#             v = self._normalize_text(pair.get('variable', '') or '')
#             words = set(q.split() + t.split() + v.split())
#             for w in words:
#                 if len(w) > 2:
#                     self.search_index[w].append(i)
#                     self.vocabulary.add(w)
#             if t:
#                 self.search_index[f'territory:{t}'].append(i)
#             if v:
#                 self.search_index[f'variable:{v}'].append(i)

#         # Construire liste normalisée des territoires
#         territories = set()
#         if hasattr(self.data_processor, 'combined_data') and self.data_processor.combined_data:
#             for item in self.data_processor.combined_data:
#                 t = item.get('territory') or item.get('original_territory') or item.get('territoire')
#                 if t:
#                     territories.add(self._normalize_text(t))
#         self.territory_list = sorted(territories)
#         self.logger.info(f'Index construit: {len(self.qa_pairs)} paires, {len(self.territory_list)} territoires')

#     # ----------------------
#     # Correction orthographique simple
#     # ----------------------

#     def _spell_correct_word(self, word: str) -> str:
#         nw = self._normalize_text(word)
#         if not nw:
#             return word
#         if nw in self.vocabulary:
#             return nw
#         if self.spell_corrections:
#             for correct, variants in self.spell_corrections.items():
#                 if nw in [self._normalize_text(v) for v in variants]:
#                     return self._normalize_text(correct)
#         # fallback fuzzy
#         if self.vocabulary:
#             matches = get_close_matches(nw, list(self.vocabulary), n=1, cutoff=0.75)
#             if matches:
#                 return matches[0]
#         return nw

#     def _spell_correct_query(self, query: str) -> str:
#         tokens = self._tokenize(query)
#         corrected = [self._spell_correct_word(t) for t in tokens]
#         return ' '.join(corrected)

#     # ----------------------
#     # Extraction territoire avancée
#     # ----------------------

#     def _normalize_territory_candidate(self, name: str) -> str:
#         """Nettoie un nom de territoire en retirant les préfixes courants et en normalisant."""
#         if not name:
#             return ''
#         n = self._normalize_text(name)
#         # Retirer "commune", "province", "prefecture", "ca", etc.
#         n = re.sub(r"\b(commune|province|prefecture|préfecture|ca|casa|municipalite|municipal)\b", '', n)
#         # Gérer les formes "d'xxx" -> garder le nom seul
#         n = re.sub(r"\bd[' ](\w[\w']*)\b", r"\1", n)
#         n = re.sub(r"\s+", ' ', n).strip()
#         return n

#     def extract_territory_from_query(self, query: str) -> Optional[str]:
#         """Extraction robuste du territoire.
#         Gère les variations telles que "d'aqsri" ou "Aqsri" quand le dataset contient "Commune d'Aqsri".
#         """
#         if not query:
#             return None

#         q_norm = self._normalize_text(query)
#         q_corrected = self._spell_correct_query(q_norm)

#         # 1) Recherche par alias statiques (si défini dans config)
#         aliases = getattr(self.config, 'TERRITORY_ALIASES', {})
#         for canonical, variants in aliases.items():
#             for v in variants:
#                 if self._normalize_text(v) in q_corrected:
#                     return canonical

#         # 2) Rechercher des tokens apparents
#         tokens = self._tokenize(q_corrected)
#         candidates = set()
#         # Si la question contient explicitement un mot ressemblant à un territoire de la liste
#         for t in self.territory_list:
#             # comparaison directe
#             if t in q_corrected:
#                 return t
#             # token overlap
#             t_tokens = set(t.split())
#             if t_tokens & set(tokens):
#                 candidates.add(t)

#         # 3) Gérer formes "d'xxx" ou "commune d'xxx" dans la requête
#         m = re.search(r"\bd[' ]([\w']{2,})\b", q_norm)
#         if m:
#             name = m.group(1)
#             name_norm = self._normalize_text(name)
#             # Chercher meilleur match par similarité dans la liste
#             matches = get_close_matches(name_norm, self.territory_list, n=1, cutoff=0.6)
#             if matches:
#                 return matches[0]
#             # essayer en enlevant/ajoutant "commune" préfix
#             matches = get_close_matches(f"commune {name_norm}", self.territory_list, n=1, cutoff=0.6)
#             if matches:
#                 return matches[0]

#         # 4) Si on a des candidats issus du token overlap, choisir le meilleur par similarité
#         if candidates:
#             # ordonner par similarité entre la requête et candidat
#             best = None
#             best_score = 0.0
#             for c in candidates:
#                 score = SequenceMatcher(None, q_corrected, c).ratio()
#                 if score > best_score:
#                     best_score = score
#                     best = c
#             if best and best_score > 0.5:
#                 return best

#         # 5) fallback: essayer get_close_matches sur la phrase complète
#         matches = get_close_matches(q_corrected, self.territory_list, n=1, cutoff=0.6)
#         if matches:
#             return matches[0]

#         return None

#     # ----------------------
#     # Extraction indicateur
#     # ----------------------

#     def extract_indicator_from_query(self, query: str) -> Optional[str]:
#         if not query:
#             return None
#         q = self._normalize_text(query)
#         q_corrected = self._spell_correct_query(q)

#         # Chercher dans mapping d'indicateurs
#         for key, variants in self.indicator_keywords.items():
#             for v in variants:
#                 if v in q_corrected:
#                     return key

#         # Tranches d'âge
#         m = re.search(r"(\d{1,2})\s*[\-–—]?\s*(\d{1,2})\s*ans", q_corrected)
#         if m:
#             return f"age_{m.group(1)}_{m.group(2)}"

#         return None

#     # ----------------------
#     # Recherche multi-niveaux (semblable à l'existant mais consolidé)
#     # ----------------------

#     def find_best_match_semantic_advanced(self, query: str) -> Optional[Dict]:
#         if not self.sentence_transformer or not self.qa_pairs:
#             return None
#         try:
#             processed_query = self.preprocess_query(query)
#             corrected_query = self._spell_correct_query(processed_query)

#             territory = self.extract_territory_from_query(corrected_query)
#             indicator = self.extract_indicator_from_query(corrected_query)

#             enriched = corrected_query
#             if territory:
#                 enriched += f" | territoire: {territory}"
#             if indicator:
#                 enriched += f" | indicateur: {indicator}"

#             q_emb = self.sentence_transformer.encode([enriched])[0]
#             q_emb = self._l2_normalize(q_emb)

#             candidates = []
#             for pair in self.qa_pairs:
#                 if 'embedding' not in pair:
#                     continue
#                 semantic_score = float(np.dot(q_emb, pair['embedding']))
#                 context_bonus = 0.0
#                 if territory and pair.get('territory'):
#                     if self._normalize_text(territory) == self._normalize_text(pair['territory']):
#                         context_bonus += 0.3
#                 if indicator and pair.get('variable'):
#                     if indicator == self._normalize_text(pair.get('variable', '')):
#                         context_bonus += 0.25
#                 final_score = semantic_score + context_bonus
#                 candidates.append((pair, final_score, semantic_score, context_bonus))

#             if not candidates:
#                 return None
#             candidates.sort(key=lambda x: x[1], reverse=True)
#             best = candidates[0]
#             if best[1] >= self.SIMILARITY_THRESHOLD:
#                 return best[0]
#             return None
#         except Exception as e:
#             self.logger.error(f"Erreur recherche sémantique: {e}")
#             return None

#     def find_best_match_index_search(self, query: str) -> Optional[Dict]:
#         if not self.search_index:
#             return None
#         try:
#             processed = self.preprocess_query(query)
#             corrected = self._spell_correct_query(processed)
#             normalized = self._normalize_text(corrected)

#             territory = self.extract_territory_from_query(corrected)
#             indicator = self.extract_indicator_from_query(corrected)

#             scores = defaultdict(float)
#             # territory exact
#             if territory:
#                 key = f"territory:{self._normalize_text(territory)}"
#                 for idx in self.search_index.get(key, []):
#                     scores[idx] += 0.5
#             # indicator exact
#             if indicator:
#                 key = f"variable:{self._normalize_text(indicator)}"
#                 for idx in self.search_index.get(key, []):
#                     scores[idx] += 0.4
#             # words
#             for w in normalized.split():
#                 if len(w) > 2 and w in self.search_index:
#                     for idx in self.search_index[w]:
#                         scores[idx] += 0.1 / max(1, len(normalized.split()))

#             if not scores:
#                 return None
#             best_idx = max(scores.keys(), key=lambda x: scores[x])
#             if scores[best_idx] >= 0.3:
#                 return self.qa_pairs[best_idx]
#             return None
#         except Exception as e:
#             self.logger.error(f"Erreur index search: {e}")
#             return None

#     def find_best_match_fuzzy_advanced(self, query: str) -> Optional[Dict]:
#         if not self.qa_pairs:
#             return None
#         try:
#             processed = self.preprocess_query(query)
#             corrected = self._spell_correct_query(processed)
#             norm = self._normalize_text(corrected)

#             best_pair = None
#             best_score = 0.0
#             for pair in self.qa_pairs:
#                 q_text = self._normalize_text(pair.get('question', ''))
#                 seq = SequenceMatcher(None, norm, q_text).ratio()
#                 # Jaccard
#                 a = set(norm.split())
#                 b = set(q_text.split())
#                 jacc = len(a & b) / len(a | b) if a or b else 0.0
#                 base = 0.6 * seq + 0.4 * jacc
#                 # contextual bonus
#                 if pair.get('territory') and self.extract_territory_from_query(corrected):
#                     if self._normalize_text(pair.get('territory')) == self._normalize_text(self.extract_territory_from_query(corrected) or ''):
#                         base += 0.25
#                 if base > best_score:
#                     best_score = base
#                     best_pair = pair
#             if best_score >= self.FUZZY_THRESHOLD:
#                 return best_pair
#             return None
#         except Exception as e:
#             self.logger.error(f"Erreur fuzzy search: {e}")
#             return None

#     # ----------------------
#     # Génération avec modèle entraîné
#     # ----------------------

#     def generate_with_trained_model(self, query: str, context: Optional[Dict] = None) -> Optional[str]:
#         if not self.is_trained or not self.model:
#             return None
#         try:
#             prompt_parts = ["Question:", query]
#             if context:
#                 if context.get('territory'):
#                     prompt_parts.append(f"Territoire: {context['territory']}")
#                 if context.get('variable'):
#                     prompt_parts.append(f"Indicateur: {context['variable']}")
#             prompt = "\n".join(prompt_parts) + "\nReponse:"

#             if getattr(self, 'nlp_pipeline', None):
#                 outputs = self.nlp_pipeline(prompt, max_new_tokens=150, do_sample=False)
#                 if outputs:
#                     text = outputs[0].get('generated_text', '')
#                     return self.clean_response(text)
#             # fallback
#             inputs = self.tokenizer.encode(prompt, return_tensors='pt')
#             if torch.cuda.is_available():
#                 inputs = inputs.cuda()
#             with torch.no_grad():
#                 out = self.model.generate(inputs, max_new_tokens=150)
#             resp = self.tokenizer.decode(out[0], skip_special_tokens=True)
#             return self.clean_response(resp)
#         except Exception as e:
#             self.logger.error(f"Erreur generation modele: {e}")
#             return None

#     # ----------------------
#     # Pipeline principal
#     # ----------------------

#     def generate_smart_response(self, query: str) -> str:
#         if self.is_greeting(query):
#             return random.choice(self.greeting_responses)

#         # 1 semantic
#         if self.sentence_transformer:
#             sem = self.find_best_match_semantic_advanced(query)
#             if sem:
#                 if self.is_trained:
#                     g = self.generate_with_trained_model(query, sem)
#                     if g:
#                         return g
#                 return sem.get('answer')
#         # 2 index
#         idx = self.find_best_match_index_search(query)
#         if idx:
#             if self.is_trained:
#                 g = self.generate_with_trained_model(query, idx)
#                 if g:
#                     return g
#             return idx.get('answer')
#         # 3 filters
#         territory = self.extract_territory_from_query(query)
#         indicator = self.extract_indicator_from_query(query)
#         f = self.search_by_filters(territory=territory, indicateur=indicator)
#         if f:
#             if self.is_trained:
#                 g = self.generate_with_trained_model(query, f)
#                 if g:
#                     return g
#             return f.get('answer')
#         # 4 fuzzy
#         fz = self.find_best_match_fuzzy_advanced(query)
#         if fz:
#             if self.is_trained:
#                 g = self.generate_with_trained_model(query, fz)
#                 if g:
#                     return g
#             return fz.get('answer')
#         # 5 model only
#         if self.is_trained:
#             m = self.generate_with_trained_model(query)
#             if m:
#                 return m
#         # guidance
#         return self.generate_guidance_response(query, territory, indicator, None)

#     # ----------------------
#     # Recherche par filtres
#     # ----------------------

#     def search_by_filters(self, territory: Optional[str] = None, question_type: Optional[str] = None,
#                           indicateur: Optional[str] = None, source: Optional[str] = None) -> Optional[Dict]:
#         if not self.qa_pairs:
#             return None
#         candidates = []
#         for pair in self.qa_pairs:
#             score = 0.0
#             match = True
#             if territory:
#                 pt = pair.get('territory', '')
#                 if self._normalize_text(territory) == self._normalize_text(pt):
#                     score += 0.4
#                 elif self._normalize_territory_candidate(self._normalize_text(territory)) in self._normalize_territory_candidate(self._normalize_text(pt)):
#                     score += 0.2
#                 else:
#                     match = False
#             if indicateur:
#                 pv = pair.get('variable', '')
#                 if self._normalize_text(indicateur) == self._normalize_text(pv):
#                     score += 0.3
#                 elif any(w in self._normalize_text(pv) for w in self._normalize_text(indicateur).split()):
#                     score += 0.15
#                 else:
#                     match = False
#             if match:
#                 candidates.append((pair, score))
#         if not candidates:
#             return None
#         candidates.sort(key=lambda x: x[1], reverse=True)
#         return candidates[0][0]

#     # ----------------------
#     # Préprocessing
#     # ----------------------

#     def preprocess_query(self, query: str) -> str:
#         if not query:
#             return ''
#         q = query.strip()
#         q = self._spell_correct_query(q)
#         replacements = {
#             r"\bmaroc\b|\broyaume du maroc\b|\bterritoire national\b|\bnational\b": 'Ensemble du territoire national',
#             r"\bpop\b|\bpopul\b": 'population',
#             r"\bmenag\b|\bmenage\b": 'ménage',
#         }
#         for pat, rep in replacements.items():
#             q = re.sub(pat, rep, q, flags=re.IGNORECASE)
#         q = re.sub(r'\s+', ' ', q).strip()
#         return q

#     def generate_guidance_response(self, query: str, territory: Optional[str], indicator: Optional[str], source: Optional[str]) -> str:
#         parts = []
#         detected = []
#         if territory:
#             detected.append(f"Territoire: {territory}")
#         if indicator:
#             detected.append(f"Indicateur: {indicator}")
#         if detected:
#             parts.append("Éléments détectés dans votre question:")
#             parts.extend([f"• {d}" for d in detected])
#             parts.append("")
#         parts.append(random.choice(["Je n'ai pas trouvé de correspondance exacte pour votre question.", "Pouvez-vous reformuler ?"]))
#         parts.append("")
#         parts.append("Exemples: \n• Population légale de la Commune d'Aqsri\n• Taille moyenne des ménages dans la province X\n• Taux de scolarisation 6-11 ans par commune")
#         if not territory:
#             parts.append("")
#             parts.append("Conseil: Précisez le territoire (région/province/commune) pour des réponses plus précises.")
#         return '\n'.join(parts)

#     def is_greeting(self, query: str) -> bool:
#         if not query:
#             return False
#         q = self._normalize_text(query)
#         greeting_patterns = ['bonjour', 'bonsoir', 'salut', 'hello', 'hi']
#         if len(q.split()) <= 3:
#             return any(p in q for p in greeting_patterns)
#         return False

#     def clean_response(self, response: str) -> str:
#         if not response or len(response.strip()) < 5:
#             return random.choice(self.default_responses)
#         s = re.sub(r"<\|[^|]+\|>", '', response)
#         s = re.sub(r'\s+', ' ', s).strip()
#         max_length = getattr(self.config, 'MAX_RESPONSE_LENGTH', 800)
#         if len(s) > max_length:
#             s = s[:max_length].rsplit(' ', 1)[0] + '...'
#         return s

#     # ----------------------
#     # Initialisation et embeddings
#     # ----------------------

#     def initialize_qa_pairs(self):
#         self.logger.info('Initialisation des paires Q&A')
#         pairs = []
#         static_faqs = getattr(self.config, 'STATIC_FAQ', [])
#         for s in static_faqs:
#             pairs.append({'question': s.get('question'), 'answer': s.get('answer'), 'territory': s.get('territory', 'Maroc'), 'variable': s.get('variable', 'info'), 'sexe': s.get('sexe', 'ensemble'), 'source_data': s.get('source', 'general')})
#         if hasattr(self.data_processor, 'combined_data') and self.data_processor.combined_data:
#             for item in self.data_processor.combined_data:
#                 q = item.get('question') or item.get('Q') or ''
#                 a = item.get('answer') or item.get('A') or item.get('value') or ''
#                 if not q or not a:
#                     continue
#                 pair = {
#                     'question': q,
#                     'answer': a,
#                     'territory': item.get('territory') or item.get('original_territory') or item.get('territoire') or 'Unknown',
#                     'variable': item.get('variable') or item.get('indicateur') or item.get('column_original') or 'unknown',
#                     'sexe': item.get('sexe') or item.get('genre') or 'ensemble',
#                     'question_type': item.get('question_type') or 'demographic',
#                     'source_data': item.get('source_data') or item.get('source') or 'non spécifié'
#                 }
#                 pairs.append(pair)
#         self.qa_pairs = pairs
#         self._build_search_index()
#         self.load_sentence_transformer()
#         if self.sentence_transformer and self.qa_pairs:
#             self._create_embeddings_with_cache()

#     def _create_embeddings_with_cache(self):
#         cache_path = getattr(self.config, 'EMBEDDINGS_CACHE_PATH', 'data/embeddings_cache.pkl')
#         try:
#             if os.path.exists(cache_path):
#                 with open(cache_path, 'rb') as f:
#                     cached = pickle.load(f)
#                 if len(cached) == len(self.qa_pairs):
#                     for i, emb in enumerate(cached):
#                         self.qa_pairs[i]['embedding'] = self._l2_normalize(np.asarray(emb, dtype=np.float32))
#                     self.logger.info('Cache embeddings chargé')
#                     return
#             # sinon calculer
#             texts = []
#             for pair in self.qa_pairs:
#                 parts = [pair.get('question', '')]
#                 if pair.get('territory'):
#                     parts.append(f"territoire: {pair.get('territory')}")
#                 if pair.get('variable'):
#                     parts.append(f"indicateur: {pair.get('variable')}")
#                 texts.append(' | '.join(parts))
#             batch = getattr(self.config, 'EMBEDDING_BATCH_SIZE', 32)
#             all_embs = []
#             for i in range(0, len(texts), batch):
#                 b = texts[i:i+batch]
#                 embs = self.sentence_transformer.encode(b, show_progress_bar=False)
#                 all_embs.extend(embs)
#             for i, emb in enumerate(all_embs):
#                 self.qa_pairs[i]['embedding'] = self._l2_normalize(np.asarray(emb, dtype=np.float32))
#             # save cache
#             try:
#                 os.makedirs(os.path.dirname(cache_path), exist_ok=True)
#                 with open(cache_path, 'wb') as f:
#                     pickle.dump(all_embs, f)
#                 self.logger.info('Cache embeddings sauvegardé')
#             except Exception as e:
#                 self.logger.warning(f"Impossible sauvegarder cache: {e}")
#         except Exception as e:
#             self.logger.error(f"Erreur creation embeddings: {e}")

#     # ----------------------
#     # Historique et statistiques
#     # ----------------------

#     def save_conversation_history(self, query: str, response: str):
#         if not getattr(self.config, 'SAVE_CONVERSATION_HISTORY', False):
#             return
#         path = getattr(self.config, 'CONVERSATION_HISTORY_PATH', 'data/conversation_history.json')
#         try:
#             os.makedirs(os.path.dirname(path), exist_ok=True)
#             history = []
#             if os.path.exists(path):
#                 with open(path, 'r', encoding='utf-8') as f:
#                     history = json.load(f)
#             entry = {
#                 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
#                 'query': query,
#                 'query_corrected': self._spell_correct_query(query),
#                 'response': response,
#                 'territory_detected': self.extract_territory_from_query(query),
#                 'indicator_detected': self.extract_indicator_from_query(query),
#                 'response_length': len(response),
#                 'model_used': 'trained' if self.is_trained else 'search_only'
#             }
#             history.append(entry)
#             max_h = getattr(self.config, 'MAX_HISTORY_SIZE', 1000)
#             history = history[-max_h:]
#             with open(path, 'w', encoding='utf-8') as f:
#                 json.dump(history, f, ensure_ascii=False, indent=2)
#         except Exception as e:
#             self.logger.error(f"Erreur sauvegarde historique: {e}")

#     def chat(self, query: str) -> str:
#         if not query or not query.strip():
#             return "Veuillez poser une question sur les statistiques démographiques ou les ménages du Maroc."
#         try:
#             self.logger.info(f"Question: {query}")
#             start = time.time()
#             r = self.generate_smart_response(query)
#             rclean = self.clean_response(r)
#             self.logger.info(f"Généré en {time.time()-start:.3f}s")
#             self.save_conversation_history(query, rclean)
#             return rclean
#         except Exception as e:
#             self.logger.error(f"Erreur chat: {e}")
#             return random.choice(self.default_responses)

#     def get_statistics(self) -> Dict:
#         stats = {
#             'is_trained': self.is_trained,
#             'qa_pairs_count': len(self.qa_pairs),
#             'has_sentence_transformer': self.sentence_transformer is not None,
#             'embedding_count': sum(1 for p in self.qa_pairs if 'embedding' in p),
#         }
#         return stats


# # EOF
















# # -*- coding: utf-8 -*-
# import os
# import re
# import json
# import time
# import logging
# import pickle
# from typing import Optional, Dict, List, Tuple, Set
# from collections import defaultdict, Counter
# from functools import lru_cache
# import unicodedata
# import random
# from dataclasses import dataclass

# import numpy as np
# import torch
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from difflib import SequenceMatcher
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel
# from sklearn.neighbors import NearestNeighbors

# # ---------------------------
# # Structures
# # ---------------------------
# @dataclass
# class SearchResult:
#     qa_pair: Dict
#     score: float
#     method: str
#     confidence: float
#     matched_fields: List[str]
#     territory_match: bool = False
#     indicator_match: bool = False

# # ---------------------------
# # Territory matcher (léger)
# # ---------------------------
# class AdvancedTerritoryMatcher:
#     def __init__(self):
#         self.territory_index = {}        # normalized -> original
#         self.territory_tokens = defaultdict(set)
#         self.normalized_cache = {}

#     @lru_cache(maxsize=8192)
#     def normalize_territory(self, territory: str) -> str:
#         if not territory:
#             return ''
#         t = territory.strip().lower()
#         t = unicodedata.normalize('NFD', t)
#         t = ''.join(ch for ch in t if unicodedata.category(ch) != 'Mn')
#         t = re.sub(r'[^\w\s\'-]', ' ', t)
#         t = re.sub(r'\s+', ' ', t).strip()
#         return t

#     def build_territory_index(self, territories: List[str]):
#         self.territory_index.clear()
#         self.territory_tokens.clear()
#         for terr in territories:
#             norm = self.normalize_territory(terr)
#             self.territory_index[norm] = terr
#             for token in norm.split():
#                 if len(token) > 2:
#                     self.territory_tokens[token].add(terr)

#     def find_territory_matches(self, query: str, threshold: float = 0.6) -> List[Tuple[str, float]]:
#         norm_q = self.normalize_territory(query)
#         if not norm_q:
#             return []
#         if norm_q in self.territory_index:
#             return [(self.territory_index[norm_q], 1.0)]
#         # token vote
#         qtok = set(norm_q.split())
#         scores = defaultdict(float)
#         for t in qtok:
#             for terr in self.territory_tokens.get(t, []):
#                 scores[terr] += 1.0 / len(qtok)
#         # fuzzy fallback limited
#         for norm_terr, orig in self.territory_index.items():
#             sim = SequenceMatcher(None, norm_q, norm_terr).ratio()
#             if sim >= threshold:
#                 scores[orig] = max(scores[orig], sim)
#         matches = sorted([(k, v) for k, v in scores.items() if v >= threshold], key=lambda x: x[1], reverse=True)
#         return matches[:5]

# # ---------------------------
# # MultiModalSearchEngine
# # ---------------------------
# class MultiModalSearchEngine:
#     def __init__(self, chatbot):
#         self.chatbot = chatbot
#         self.tfidf_vectorizer = None
#         self.tfidf_matrix = None
#         self.territory_matcher = AdvancedTerritoryMatcher()
#         self.nn_index = None     # NearestNeighbors for embeddings

#     def build_comprehensive_index(self):
#         """Construit TF-IDF et index territorial (appelé après qa_pairs préparées)."""
#         if not self.chatbot.qa_pairs:
#             return

#         # TF-IDF documents (déjà normalisés dans qa_pairs)
#         docs = [pair.get('_tfidf_document', '') for pair in self.chatbot.qa_pairs]
#         max_feat = getattr(self.chatbot.config, 'TFIDF_MAX_FEATURES', 3000)
#         try:
#             self.tfidf_vectorizer = TfidfVectorizer(max_features=max_feat, ngram_range=(1,3), lowercase=True)
#             self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(docs)
#             self.chatbot.logger.info(f"TF-IDF construit shape={self.tfidf_matrix.shape}")
#         except Exception as e:
#             self.chatbot.logger.error(f"Erreur TF-IDF: {e}")
#             self.tfidf_vectorizer = None
#             self.tfidf_matrix = None

#         # territory index
#         territories = list({pair.get('_norm_territory', '') for pair in self.chatbot.qa_pairs if pair.get('_norm_territory')})
#         self.territory_matcher.build_territory_index(territories)

#         # semantic nn index (if embeddings exist)
#         emb_list = [pair.get('embedding') for pair in self.chatbot.qa_pairs if pair.get('embedding') is not None]
#         if emb_list:
#             X = np.vstack(emb_list).astype(np.float32)
#             try:
#                 n_neighbors = min(10, X.shape[0])
#                 self.nn_index = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='auto').fit(X)
#                 self.chatbot.logger.info("NearestNeighbors index construit pour embeddings")
#             except Exception as e:
#                 self.chatbot.logger.warning(f"Impossible de construire NN index: {e}")
#                 self.nn_index = None

#     def search_tfidf(self, query: str, top_k: int = 8) -> List[SearchResult]:
#         if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
#             return []
#         normalized_query = self.chatbot._normalize_text(query)
#         qvec = self.tfidf_vectorizer.transform([normalized_query])
#         # linear_kernel (fast for sparse)
#         sim = linear_kernel(qvec, self.tfidf_matrix).flatten()
#         if sim.sum() == 0:
#             return []
#         top_idx = np.argsort(sim)[::-1][:top_k]
#         results = []
#         for idx in top_idx:
#             score = float(sim[idx])
#             if score > 0.05:
#                 results.append(SearchResult(
#                     qa_pair=self.chatbot.qa_pairs[int(idx)],
#                     score=score,
#                     method='tfidf',
#                     confidence=min(score * 1.3, 1.0),
#                     matched_fields=['tfidf_global']
#                 ))
#         return results

#     def search_semantic(self, query: str, top_k: int = 6, territory: str = None, indicator: str = None) -> List[SearchResult]:
#         if not self.chatbot.sentence_transformer or 'embedding' not in self.chatbot.qa_pairs[0]:
#             return []
#         enriched = query
#         if territory:
#             enriched += f" | territoire: {territory}"
#         if indicator:
#             enriched += f" | indicateur: {indicator}"
#         q_emb = self.chatbot.sentence_transformer.encode([enriched], convert_to_numpy=True, show_progress_bar=False)[0]
#         q_emb = self.chatbot._l2_normalize(q_emb.astype(np.float32))
#         # use NN index if exists
#         results = []
#         try:
#             if self.nn_index is not None:
#                 distances, indices = self.nn_index.kneighbors([q_emb], n_neighbors=min(top_k, self.nn_index.n_neighbors))
#                 # distances are cosine distances (0..2), convert
#                 scores = 1 - distances.flatten()
#                 for idx, sc in zip(indices.flatten(), scores):
#                     final_score = float(sc)
#                     if final_score >= self.chatbot.SIMILARITY_THRESHOLD * 0.5:
#                         results.append(SearchResult(
#                             qa_pair=self.chatbot.qa_pairs[int(idx)],
#                             score=final_score,
#                             method='semantic',
#                             confidence=min(final_score * 1.05, 1.0),
#                             matched_fields=['semantic']
#                         ))
#                 results.sort(key=lambda r: r.score, reverse=True)
#                 return results[:top_k]
#             else:
#                 # fallback brute-force vector dot (should be rare)
#                 for i, pair in enumerate(self.chatbot.qa_pairs):
#                     emb = pair.get('embedding')
#                     if emb is None:
#                         continue
#                     sc = float(np.dot(q_emb, emb))
#                     if sc >= self.chatbot.SIMILARITY_THRESHOLD * 0.6:
#                         results.append(SearchResult(pair, sc, 'semantic', min(sc*1.05,1.0), ['semantic']))
#                 results.sort(key=lambda r: r.score, reverse=True)
#                 return results[:top_k]
#         except Exception as e:
#             self.chatbot.logger.debug(f"Erreur semantic search NN: {e}")
#             return []

#     def field_specific_candidates(self, tokens: Set[str], top_limit: int = 500) -> Set[int]:
#         """Récupère ensemble d'indices candidats via index inversé."""
#         idxs = set()
#         for t in tokens:
#             postings = self.chatbot.search_index.get(t)
#             if postings:
#                 # postings is a set
#                 idxs.update(postings)
#             # also try territory/variable keys
#             tk = f"territory:{t}"
#             if tk in self.chatbot.search_index:
#                 idxs.update(self.chatbot.search_index[tk])
#             vk = f"variable:{t}"
#             if vk in self.chatbot.search_index:
#                 idxs.update(self.chatbot.search_index[vk])
#             if len(idxs) >= top_limit:
#                 break
#         return idxs

#     def search_field_specific(self, query: str, territory: str = None, indicator: str = None, top_k: int = 8) -> List[SearchResult]:
#         norm_q = self.chatbot._normalize_text(query)
#         q_tokens = set(norm_q.split())
#         candidates = self.field_specific_candidates(q_tokens, top_limit=getattr(self.chatbot.config,'CANDIDATE_LIMIT',500))
#         results = []
#         # evaluate only candidates
#         for idx in candidates:
#             pair = self.chatbot.qa_pairs[int(idx)]
#             score = 0.0
#             matched = []
#             # territory matching quick check via normalized fields
#             if territory:
#                 if pair.get('_norm_territory') and pair['_norm_territory'] == self.territory_matcher.normalize_territory(territory):
#                     score += 0.45
#                     matched.append('territory_exact')
#             # indicator similarity
#             if indicator:
#                 ind_sim = self.chatbot._calculate_indicator_similarity(indicator, pair.get('variable',''))
#                 if ind_sim > 0.3:
#                     score += 0.25 * ind_sim
#                     matched.append('indicator')
#             # question similarity quick Jaccard
#             set_q = q_tokens
#             set_p = pair.get('_norm_tokens', set())
#             if set_q and set_p:
#                 j = len(set_q & set_p) / len(set_q | set_p)
#                 score += 0.4 * j
#                 if j > 0.2:
#                     matched.append('question_jaccard')
#             if score > 0.15:
#                 results.append(SearchResult(pair, score, 'field_specific', min(score*1.2,1.0), matched))
#         results.sort(key=lambda r: r.score, reverse=True)
#         return results[:top_k]

#     def fuzzy_search(self, query: str, top_k: int = 6) -> List[SearchResult]:
#         norm_q = self.chatbot._normalize_text(query)
#         q_tokens = set(norm_q.split())
#         candidates = self.field_specific_candidates(q_tokens, top_limit=getattr(self.chatbot.config,'CANDIDATE_LIMIT',300))
#         results = []
#         for idx in candidates:
#             pair = self.chatbot.qa_pairs[int(idx)]
#             q_norm = norm_q
#             p_norm = pair.get('_norm_question', '')
#             seq_sim = SequenceMatcher(None, q_norm, p_norm).ratio()
#             set_p = pair.get('_norm_tokens', set())
#             j = len(q_tokens & set_p) / len(q_tokens | set_p) if q_tokens or set_p else 0.0
#             combined = 0.6 * seq_sim + 0.4 * j
#             if combined >= getattr(self.chatbot, 'FUZZY_THRESHOLD', 0.60):
#                 results.append(SearchResult(pair, combined, 'fuzzy', min(combined*0.9,1.0), ['question_fuzzy']))
#         results.sort(key=lambda r: r.score, reverse=True)
#         return results[:top_k]

# # ---------------------------
# # Chatbot main class (optimisé)
# # ---------------------------
# class HCPChatbotOptimized:
#     def __init__(self, config, data_processor):
#         self.config = config
#         self.data_processor = data_processor
#         logging.basicConfig(level=getattr(config, 'LOG_LEVEL', logging.INFO))
#         self.logger = logging.getLogger('HCPChatbotOptimized')

#         self.qa_pairs: List[Dict] = []
#         self.is_trained = False
#         self.sentence_transformer: Optional[SentenceTransformer] = None
#         self.search_engine = MultiModalSearchEngine(self)
#         self.performance_stats = {'search_times': [], 'method_usage': Counter(), 'territory_matches': 0, 'indicator_matches': 0}

#         # inverted index token -> set(indices)
#         self.search_index: Dict[str, Set[int]] = defaultdict(set)
#         self.vocabulary: Set[str] = set()
#         self.territory_list: List[str] = []

#         # thresholds / tunables
#         self.SIMILARITY_THRESHOLD = getattr(self.config, 'SIMILARITY_THRESHOLD', 0.65)
#         self.FUZZY_THRESHOLD = getattr(self.config, 'FUZZY_THRESHOLD', 0.60)

#         # defaults
#         self.default_responses = getattr(config, 'DEFAULT_RESPONSES', [
#             "Je ne trouve pas d'informations précises sur cette question dans ma base de données HCP.",
#             "Pouvez-vous reformuler votre question ou être plus spécifique ?",
#             "Cette information n'est pas disponible dans mes données actuelles."
#         ])
#         self.greeting_responses = getattr(config, 'GREETING_RESPONSES', [
#             "Bonjour ! Je suis l'assistant statistique du HCP. Comment puis-je vous aider ?",
#             "Salut ! Je peux vous renseigner sur les statistiques démographiques du Maroc.",
#             "Bonjour ! Posez-moi vos questions sur la population et les ménages marocains."
#         ])

#     # ---------------------------
#     # Normalisation (cached)
#     # ---------------------------
#     @lru_cache(maxsize=16384)
#     def _normalize_text(self, text: str) -> str:
#         if not text:
#             return ''
#         t = text.strip().lower()
#         t = unicodedata.normalize('NFD', t)
#         t = ''.join(ch for ch in t if unicodedata.category(ch) != 'Mn')
#         t = re.sub(r"[^\w\s'-]", ' ', t)
#         t = re.sub(r'\s+', ' ', t).strip()
#         return t

#     @staticmethod
#     def _l2_normalize(v: np.ndarray) -> np.ndarray:
#         v = np.asarray(v, dtype=np.float32)
#         n = np.linalg.norm(v)
#         return v / (n + 1e-8)

#     # ---------------------------
#     # Préparation des paires (précompute)
#     # ---------------------------
#     def initialize_qa_pairs(self):
#         self.logger.info("Initialisation QA pairs et indexation précomputée...")
#         pairs = []

#         # static faqs
#         static_faqs = getattr(self.config, 'STATIC_FAQ', [])
#         for s in static_faqs:
#             pairs.append({
#                 'question': s.get('question') or '',
#                 'answer': s.get('answer') or '',
#                 'territory': s.get('territory', 'Maroc'),
#                 'variable': s.get('variable', 'info'),
#                 'sexe': s.get('sexe', 'ensemble'),
#                 'source_data': s.get('source','static')
#             })

#         # data processor combined_data
#         if hasattr(self.data_processor, 'combined_data') and self.data_processor.combined_data:
#             for item in self.data_processor.combined_data:
#                 q = item.get('question') or item.get('Q') or ''
#                 a = item.get('answer') or item.get('A') or item.get('value') or ''
#                 if not q or a is None:
#                     continue
#                 pairs.append({
#                     'question': q,
#                     'answer': a,
#                     'territory': item.get('territory') or item.get('original_territory') or item.get('territoire') or 'Unknown',
#                     'variable': item.get('variable') or item.get('indicateur') or item.get('column_original') or 'unknown',
#                     'sexe': item.get('sexe') or item.get('genre') or 'ensemble',
#                     'source_data': item.get('source_data') or item.get('source') or 'non spécifié'
#                 })

#         # store
#         self.qa_pairs = pairs
#         self.logger.info(f"Chargé {len(self.qa_pairs)} paires Q&A")

#         # Precompute normalized fields and inverted index
#         for i, pair in enumerate(self.qa_pairs):
#             q = pair.get('question','')
#             norm_q = self._normalize_text(q)
#             pair['_norm_question'] = norm_q
#             tokens = {w for w in norm_q.split() if len(w)>2}
#             pair['_norm_tokens'] = tokens
#             pair['_norm_territory'] = self._normalize_text(pair.get('territory',''))
#             pair['_norm_variable'] = self._normalize_text(pair.get('variable',''))
#             # TF-IDF document (concatenate relevant fields)
#             parts = [pair.get('question',''), pair.get('territory',''), pair.get('variable',''), str(pair.get('answer',''))]
#             pair['_tfidf_document'] = self._normalize_text(' '.join([p for p in parts if p]))
#             # fill inverted index
#             for token in tokens:
#                 self.search_index[token].add(i)
#                 self.vocabulary.add(token)
#             # field-specific keys
#             if pair['_norm_territory']:
#                 self.search_index[f"territory:{pair['_norm_territory']}"].add(i)
#             if pair['_norm_variable']:
#                 self.search_index[f"variable:{pair['_norm_variable']}"].add(i)

#         # build territory list
#         self.territory_list = sorted({p.get('territory','') for p in self.qa_pairs if p.get('territory') and p.get('territory')!='Unknown'})

#         # load embedding model (if available) and TFIDF + other indexes
#         self.load_sentence_transformer()
#         # embeddings creation (from cache if possible)
#         if self.sentence_transformer:
#             self._create_embeddings_with_cache()
#         # build comprehensive index in search engine
#         self.search_engine.build_comprehensive_index()

#     # ---------------------------
#     # Embeddings (cache + batch)
#     # ---------------------------
#     def load_sentence_transformer(self):
#         model_name = getattr(self.config, 'EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
#         try:
#             self.sentence_transformer = SentenceTransformer(model_name)
#             self.sentence_transformer.eval()
#             if torch.cuda.is_available():
#                 try:
#                     self.sentence_transformer = self.sentence_transformer.cuda()
#                     self.logger.info("SentenceTransformer sur GPU")
#                 except Exception:
#                     self.logger.warning("Impossible de placer transformer sur GPU")
#             self.logger.info(f"SentenceTransformer {model_name} chargé")
#         except Exception as e:
#             self.logger.error(f"Erreur chargement SentenceTransformer: {e}")
#             self.sentence_transformer = None

#     def _create_embeddings_with_cache(self):
#         cache_path = getattr(self.config, 'EMBEDDINGS_CACHE_PATH', 'data/embeddings_cache.pkl')
#         texts = []
#         for pair in self.qa_pairs:
#             parts = [pair.get('question',''), f"territoire: {pair.get('territory','')}", f"indicateur: {pair.get('variable','')}"]
#             texts.append(' | '.join([p for p in parts if p]))

#         # try load cache
#         loaded = False
#         if os.path.exists(cache_path):
#             try:
#                 with open(cache_path,'rb') as f:
#                     cache = pickle.load(f)
#                 if isinstance(cache, dict) and cache.get('count')==len(texts) and cache.get('model')==getattr(self.config,'EMBEDDING_MODEL',None):
#                     emb = cache['embeddings']
#                     for i, e in enumerate(emb):
#                         self.qa_pairs[i]['embedding'] = self._l2_normalize(np.asarray(e, dtype=np.float32))
#                     loaded = True
#                     self.logger.info(f"Embeddings chargés depuis cache ({len(emb)})")
#             except Exception as e:
#                 self.logger.warning(f"Erreur lecture cache embeddings: {e}")

#         if not loaded:
#             self.logger.info("Génération des embeddings (batch)...")
#             batch_size = getattr(self.config, 'EMBEDDING_BATCH_SIZE', 64)
#             all_emb = []
#             for i in range(0, len(texts), batch_size):
#                 batch = texts[i:i+batch_size]
#                 emb = self.sentence_transformer.encode(batch, convert_to_numpy=True, show_progress_bar=False)
#                 for e in emb:
#                     all_emb.append(self._l2_normalize(e.astype(np.float32)))
#             # attach
#             for i, e in enumerate(all_emb):
#                 self.qa_pairs[i]['embedding'] = e
#             # save
#             try:
#                 os.makedirs(os.path.dirname(cache_path), exist_ok=True)
#                 cache_data = {'embeddings': all_emb, 'count': len(all_emb), 'model': getattr(self.config,'EMBEDDING_MODEL',None)}
#                 with open(cache_path,'wb') as f:
#                     pickle.dump(cache_data, f)
#                 self.logger.info(f"Embeddings sauvegardés: {len(all_emb)}")
#             except Exception as e:
#                 self.logger.warning(f"Impossible sauver cache embeddings: {e}")

#     # ---------------------------
#     # Extraction helper
#     # ---------------------------
#     def advanced_extract_territory(self, query: str) -> Optional[str]:
#         if not query:
#             return None
#         matches = self.search_engine.territory_matcher.find_territory_matches(query)
#         return matches[0][0] if matches else None

#     def advanced_extract_indicator(self, query: str) -> Tuple[Optional[str], Optional[str]]:
#         if not query:
#             return None, None
#         nq = self._normalize_text(query)
#         best_match = None
#         best_domain = None
#         best_score = 0.0
#         # simple scan of indicator_domains from config or internal mapping
#         indicator_domains = getattr(self.config, 'INDICATOR_DOMAINS', {})
#         for domain, cats in indicator_domains.items():
#             for cat, inds in cats.items():
#                 for ind in inds:
#                     nind = self._normalize_text(ind)
#                     if nind in nq:
#                         return ind, domain
#                     tokens_ind = set(nind.split())
#                     if tokens_ind:
#                         score = len(tokens_ind & set(nq.split())) / len(tokens_ind)
#                         if score > best_score:
#                             best_score = score
#                             best_match = ind
#                             best_domain = domain
#         if best_score > 0.6:
#             return best_match, best_domain
#         return None, None

#     # ---------------------------
#     # Unified search orchestration
#     # ---------------------------
#     def unified_search(self, query: str) -> List[SearchResult]:
#         start = time.time()
#         results = []
#         territory = self.advanced_extract_territory(query)
#         indicator, domain = self.advanced_extract_indicator(query)

#         # 1 Semantic (fast via NN)
#         try:
#             sem = self.search_engine.search_semantic(query, top_k=6, territory=territory, indicator=indicator)
#             results.extend(sem)
#             self.performance_stats['method_usage']['semantic'] += 1
#         except Exception as e:
#             self.logger.debug(f"Semantic search error: {e}")

#         # 2 TF-IDF
#         try:
#             tf = self.search_engine.search_tfidf(query, top_k=6)
#             results.extend(tf)
#             self.performance_stats['method_usage']['tfidf'] += 1
#         except Exception as e:
#             self.logger.debug(f"TF-IDF search error: {e}")

#         # 3 field specific (limited candidate set)
#         try:
#             fs = self.search_engine.search_field_specific(query, territory=territory, indicator=indicator, top_k=8)
#             results.extend(fs)
#             self.performance_stats['method_usage']['field_specific'] += 1
#         except Exception as e:
#             self.logger.debug(f"Field specific error: {e}")

#         # 4 fuzzy fallback
#         try:
#             fz = self.search_engine.fuzzy_search(query, top_k=6)
#             results.extend(fz)
#             self.performance_stats['method_usage']['fuzzy'] += 1
#         except Exception as e:
#             self.logger.debug(f"Fuzzy search error: {e}")

#         # merge/dedupe
#         merged = self._merge_and_deduplicate_results(results)
#         elapsed = time.time() - start
#         self.performance_stats['search_times'].append(elapsed)
#         if territory: self.performance_stats['territory_matches'] += 1
#         if indicator: self.performance_stats['indicator_matches'] += 1
#         return merged[:10]

#     # ---------------------------
#     # Merge/dedupe
#     # ---------------------------
#     def _merge_and_deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
#         seen = {}
#         merged = []
#         for r in results:
#             pid = f"{r.qa_pair.get('question','')[:80]}_{r.qa_pair.get('territory','')}"
#             if pid in seen:
#                 ex = seen[pid]
#                 ex.score = max(ex.score, r.score)
#                 ex.confidence = max(ex.confidence, r.confidence)
#                 ex.matched_fields = list(set(ex.matched_fields + r.matched_fields))
#                 ex.method = ex.method + "+" + r.method if r.method not in ex.method else ex.method
#             else:
#                 seen[pid] = r
#                 merged.append(r)
#         merged.sort(key=lambda x: (x.score, x.confidence), reverse=True)
#         return merged

#     # ---------------------------
#     # Response pipeline
#     # ---------------------------
#     def generate_smart_response(self, query: str) -> str:
#         if self.is_greeting(query):
#             return random.choice(self.greeting_responses)
#         results = self.unified_search(query)
#         if results:
#             best = results[0]
#             # try model generation if available
#             if self.is_trained:
#                 gen = self.generate_with_trained_model(query, best.qa_pair)
#                 if gen:
#                     return gen
#             # fallback to direct answer
#             return str(best.qa_pair.get('answer',''))
#         # fallback guidance
#         territory = self.advanced_extract_territory(query)
#         indicator, domain = self.advanced_extract_indicator(query)
#         return self.generate_guidance_response(query, territory, indicator, domain)

#     def generate_with_trained_model(self, query: str, context: Optional[Dict] = None) -> Optional[str]:
#         # keep original logic but it's rarely used in search-only mode
#         return None  # placeholder - keep your existing implementation if needed

#     def is_greeting(self, query: str) -> bool:
#         if not query: return False
#         norm = self._normalize_text(query)
#         greetings = ['bonjour','bonsoir','salut','hello','hi','hey']
#         if len(norm.split()) <= 4:
#             return any(g in norm for g in greetings)
#         return False

#     def clean_response(self, response: str) -> str:
#         if not response or len(response.strip()) < 5:
#             return random.choice(self.default_responses)
#         cleaned = re.sub(r"<\|[^|]+\|>", '', response)
#         cleaned = re.sub(r'\[.*?\]', '', cleaned)
#         cleaned = re.sub(r'\s+', ' ', cleaned).strip()
#         maxl = getattr(self.config, 'MAX_RESPONSE_LENGTH', 800)
#         if len(cleaned) > maxl:
#             cleaned = cleaned[:maxl].rsplit(' ',1)[0] + '...'
#         if len(cleaned) < 10:
#             return random.choice(self.default_responses)
#         return cleaned

#     def generate_guidance_response(self, query, territory, indicator, domain):
#         parts = []
#         if territory: parts.append(f"Territoire détecté: {territory}")
#         if indicator: parts.append(f"Indicateur détecté: {indicator}")
#         if parts:
#             parts.append("")
#         parts.append("Je n'ai pas trouvé de correspondance exacte pour cette question dans mes données.")
#         parts.append("Exemples de requêtes valides: ...")
#         return "\n".join(parts)

#     def chat(self, query: str) -> str:
#         if not query or not query.strip():
#             return "Veuillez poser une question sur les statistiques démographiques ou les ménages du Maroc."
#         try:
#             start = time.time()
#             resp = self.generate_smart_response(query)
#             elapsed = time.time() - start
#             self.logger.info(f"Réponse générée en {elapsed:.3f}s")
#             # optionally save history (kept out for perf unless enabled)
#             return self.clean_response(resp)
#         except Exception as e:
#             self.logger.error(f"Erreur chat: {e}")
#             return random.choice(self.default_responses)

#     # metrics helpers
#     def get_performance_report(self):
#         st = self.performance_stats['search_times']
#         return {
#             'avg_ms': round(np.mean(st)*1000,2) if st else None,
#             'count': len(st),
#             'method_usage': dict(self.performance_stats['method_usage'])
#         }

# # End of file

