"""
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
