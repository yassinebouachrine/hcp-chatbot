import json
import pandas as pd
import numpy as np
import re
import os
import hashlib
from sentence_transformers import SentenceTransformer
import faiss


class HCPDataProcessor:
    """Processeur HCP adapté à la nouvelle structure JSON (qa_pairs) et à divers cas limites.

    Améliorations clés :
    - Support robuste des clés multilingues/variants ("territoire" / "territory") via NEW_DATA_STRUCTURE si fourni dans config
    - Lecture des métadonnées (statistics, total_qa, etc.) quand disponibles
    - Extraction numérique étendue (nombres purs, séparateurs français et anglais, pourcentages)
    - Déduplication stable via SHA256 plutôt que hash() volatile
    - Robustesse de la création d'embeddings (assure np.array float32) et création d'index FAISS
    - Mapping de "column_original" vers variable si disponible
    """

    def __init__(self, config):
        self.config = config
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.data = None
        self.raw_data = None
        self.embeddings = None
        self.index = None
        self.combined_data = []

        # Territoires du Maroc pour normalisation (rajouter variantes si besoin)
        self.morocco_territories = {
            'maroc': 'Ensemble du territoire national',
            'royaume du maroc': 'Ensemble du territoire national',
            'ensemble du territoire national': 'Ensemble du territoire national',
            'territoire national': 'Ensemble du territoire national',
            'national': 'Ensemble du territoire national'
        }

        # Récupérer les noms de champs personnalisés depuis la config si présent
        self.root_key = getattr(config, 'NEW_DATA_STRUCTURE', {}).get('root_key', 'qa_pairs')
        self.q_key = getattr(config, 'NEW_DATA_STRUCTURE', {}).get('question_key', 'question')
        self.a_key = getattr(config, 'NEW_DATA_STRUCTURE', {}).get('answer_key', 'answer')
        self.t_key = getattr(config, 'NEW_DATA_STRUCTURE', {}).get('territory_key', 'territoire')
        self.ind_key = getattr(config, 'NEW_DATA_STRUCTURE', {}).get('indicator_key', 'indicateur')
        self.gender_key = getattr(config, 'NEW_DATA_STRUCTURE', {}).get('gender_key', 'genre')
        self.source_key = getattr(config, 'NEW_DATA_STRUCTURE', {}).get('source_key', 'source')

    def load_all_data(self) -> pd.DataFrame:
        """Charge les données depuis les chemins configurés en supportant la nouvelle structure.
        Retourne un DataFrame pandas.
        """
        all_data = []

        for data_type, file_path in self.config.DATA_PATHS.items():
            if os.path.exists(file_path):
                print(f"Chargement de {file_path} ({data_type})...")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)

                    # Traiter la structure détectée
                    processed_data = self._process_new_structure_data(file_data, data_type)
                    all_data.extend(processed_data)
                    print(f"  ✓ {len(processed_data)} entrées chargées et validées")

                except Exception as e:
                    print(f"  ✗ Erreur lors du chargement de {file_path}: {e}")

        # Fallback vers l'ancien fichier si nécessaire
        if not all_data and hasattr(self.config, 'DATA_PATH') and os.path.exists(self.config.DATA_PATH):
            print(f"Tentative de chargement du fichier legacy: {self.config.DATA_PATH}")
            try:
                with open(self.config.DATA_PATH, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                processed_data = self._process_new_structure_data(file_data, 'indicators')
                all_data.extend(processed_data)
                print(f"  ✓ {len(processed_data)} entrées chargées depuis le fichier legacy")
            except Exception as e:
                print(f"  ✗ Erreur lors du chargement du fichier legacy: {e}")

        if not all_data:
            print("❌ Aucune donnée chargée. Vérifiez vos fichiers JSON.")
            return pd.DataFrame()

        # Validation et nettoyage
        valid_data = self._validate_loaded_data(all_data)
        self.data = pd.DataFrame(valid_data)
        self.combined_data = valid_data

        print(f"✅ Total: {len(self.data)} paires question-réponse validées")
        print(f"Colonnes: {list(self.data.columns)}")

        return self.data

    def _process_new_structure_data(self, data, data_type: str):
        """Traite les données avec la structure moderne (qa_pairs) ou legacy.
        Compatible avec différentes variantes de clés.
        """
        processed_rows = []

        # Assurer structure 'qa_pairs'
        if isinstance(data, dict) and (self.root_key in data or 'qa_pairs' in data):
            root = data.get(self.root_key, data.get('qa_pairs', []))
            print(f"  Structure moderne détectée: {len(root)} éléments dans {self.root_key}")

            # Extraire des métadonnées utiles si présentes
            metadata = data.get('metadata') or {}
            # essayer d'extraire un total depuis metadata.statistics ou champs standards
            meta_total = None
            if isinstance(metadata, dict):
                meta_total = metadata.get('total_qa_pairs') or metadata.get('total_qa') or (
                    metadata.get('statistics', {}).get('total_qa') if isinstance(metadata.get('statistics'), dict) else None
                )
                if meta_total:
                    print(f"  Métadonnées: total_qa ~ {meta_total}")

            valid_count = 0
            for item in root:
                if not isinstance(item, dict):
                    continue

                # Extraire champs selon config/fallbacks
                question = (item.get(self.q_key) or item.get('question') or '').strip()
                answer = (item.get(self.a_key) or item.get('answer') or
                          item.get('reponse') or item.get('réponse') or '').strip()

                # Territory variants
                territoire = (item.get(self.t_key) or item.get('territoire') or item.get('territory') or '').strip()
                indicateur = (item.get(self.ind_key) or item.get('indicateur') or item.get('ind') or '').strip()
                genre = (item.get(self.gender_key) or item.get('genre') or item.get('sexe') or '').strip()
                source = (item.get(self.source_key) or item.get('source') or item.get('source_data') or '').strip()
                column_original = item.get('column_original') or item.get('column') or ''

                if not question or not answer:
                    continue

                valid_count += 1

                # Normaliser territoire
                territory_normalized = self._normalize_territory(territoire)

                # Classifier automatiquement le type de question
                question_type = self._classify_question_from_indicateur(indicateur, question, answer)

                # Extraire indicateurs numériques
                indicators = self._extract_numerical_indicators_from_text(answer)

                # Utiliser column_original pour déterminer variable si indicateur absent
                variable = indicateur or self._infer_variable_from_column(column_original)
                if not variable:
                    variable = self._extract_variable_from_text(question, answer)

                # Construire ID stable (SHA256) pour dedup
                q_norm = self._clean_text(question).lower()
                q_hash = hashlib.sha256(q_norm.encode('utf-8')).hexdigest()

                processed_row = {
                    'question': question,
                    'answer': answer,
                    'territory': territory_normalized,
                    'original_territory': territoire,
                    'question_type': question_type,
                    'indicators': indicators,
                    'data_type': data_type,
                    'sexe': genre if genre else 'ensemble',
                    'variable': variable if variable else 'indicateur_demographique',
                    'source_data': source if source else data_type,
                    'column_original': column_original,
                    'description': f"Question {question_type} sur {territory_normalized}",
                    'question_hash': q_hash,
                    'data_source': 'nouvelle_structure'
                }

                processed_rows.append(processed_row)

            print(f"  ✓ {valid_count} éléments traités avec succès (nouvelle structure)")

        else:
            # Fallback legacy
            print(f"  Structure legacy détectée ou racine introuvable, tentative de traitement legacy...")
            processed_rows = self._process_legacy_data(data, data_type)

        # Dedup stable par question_hash
        unique_rows = {}
        for row in processed_rows:
            q_hash = row['question_hash']
            if q_hash not in unique_rows:
                unique_rows[q_hash] = row

        final_rows = list(unique_rows.values())
        if len(final_rows) != len(processed_rows):
            print(f"  ℹ️ {len(processed_rows) - len(final_rows)} doublons supprimés")

        return final_rows

    def _infer_variable_from_column(self, column_original: str) -> str:
        """Tente de mapper column_original vers une clé d'indicateur via la config HCP_INDICATOR_MAPPING
        (ative si la colonne correspond à un label mappé).
        """
        if not column_original:
            return ''
        col = column_original.lower()
        # Chercher une correspondance simple
        for key, label in getattr(self.config, 'HCP_INDICATOR_MAPPING', {}).items():
            if label.lower() in col or key.replace('_', ' ') in col:
                return key
        # heuristique: retirer préfixes comme 'sexe :' et chercher mots clefs
        cleaned = re.sub(r"[^a-z0-9\s]", ' ', col)
        tokens = cleaned.split()
        # check some tokens
        token_map = {
            'légale': 'population_legale',
            'municipale': 'population_municipale',
            'ménages': 'menage_taille',
            'taux': 'percentage'
        }
        for t in tokens:
            if t in token_map:
                return token_map[t]
        return ''

    def _classify_question_from_indicateur(self, indicateur: str, question: str, answer: str) -> str:
        """Classification des types de questions basée sur l'indicateur (si disponible) sinon fallback textuel."""
        if not indicateur:
            return self._classify_question_from_text(question, answer)

        indicateur_lower = indicateur.lower()

        # Classification basée sur mapping plus robuste
        if 'population_legale' in indicateur_lower or 'légale' in indicateur_lower:
            return 'population_legale'
        elif 'population_municipale' in indicateur_lower or 'municipale' in indicateur_lower:
            return 'population_municipale'
        elif 'masculin' in indicateur_lower or 'pourcentage_masculin' in indicateur_lower:
            return 'demographics_gender'
        elif 'féminin' in indicateur_lower or 'pourcentage_feminin' in indicateur_lower:
            return 'demographics_gender'
        elif 'matrimonial' in indicateur_lower or 'mari' in indicateur_lower:
            return 'marital_status'
        elif 'age' in indicateur_lower or 'tranche' in indicateur_lower:
            return 'demographics_age'
        elif 'emploi' in indicateur_lower or 'chomage' in indicateur_lower:
            return 'employment'
        elif 'education' in indicateur_lower or 'scolarisation' in indicateur_lower:
            return 'education'
        elif 'logement' in indicateur_lower or 'menage' in indicateur_lower:
            return 'housing'
        elif 'population' in indicateur_lower:
            return 'population_count'
        else:
            return 'demographic_general'

    def _process_legacy_data(self, data, data_type: str):
        """Traite structure legacy et retente d'extraire champs connus."""
        processed_rows = []
        items_to_process = []

        if isinstance(data, dict):
            # Quand le dict est mappage d'items
            for key, value in data.items():
                if isinstance(value, dict) and any(k in value for k in ['question', 'question_text']):
                    # Normaliser answer key
                    for key_variant in ['response', 'reponse', 'answer', 'réponse']:
                        if key_variant in value:
                            value['answer'] = value[key_variant]
                            break
                    items_to_process.append(value)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and any(k in item for k in ['question', 'question_text']):
                    for key_variant in ['response', 'reponse', 'réponse', 'answer']:
                        if key_variant in item:
                            item['answer'] = item[key_variant]
                            break
                    items_to_process.append(item)

        print(f"  Traitement legacy de {len(items_to_process)} éléments...")

        valid_count = 0
        for item in items_to_process:
            if not isinstance(item, dict):
                continue

            question = (item.get('question') or item.get('question_text') or '').strip()
            answer = (item.get('answer') or '').strip()

            if not question or not answer:
                continue

            valid_count += 1

            territory = self._extract_territory_from_text(question, answer)
            question_type = self._classify_question_from_text(question, answer)
            indicators = self._extract_numerical_indicators_from_text(answer)
            sexe = self._extract_gender_from_text(question, answer)
            variable = self._extract_variable_from_text(question, answer)

            q_hash = hashlib.sha256(self._clean_text(question).lower().encode('utf-8')).hexdigest()

            processed_row = {
                'question': question,
                'answer': answer,
                'territory': territory,
                'original_territory': territory,
                'question_type': question_type,
                'indicators': indicators,
                'data_type': data_type,
                'sexe': sexe,
                'variable': variable,
                'source_data': 'legacy',
                'description': f"Question {question_type} sur {territory}",
                'question_hash': q_hash,
                'data_source': 'legacy_structure'
            }

            processed_rows.append(processed_row)

        print(f"  ✓ {valid_count} éléments traités avec succès (structure legacy)")
        return processed_rows

    def _normalize_territory(self, territoire: str) -> str:
        """Normalise le nom du territoire selon nos standards."""
        if not territoire:
            return 'Territoire non spécifié'

        territoire_lower = territoire.lower().strip()

        # Vérifier correspondances exactes
        for territory_variant, standard_name in self.morocco_territories.items():
            if territory_variant == territoire_lower:
                return standard_name

        # Correspondances partielles
        for territory_variant, standard_name in self.morocco_territories.items():
            if territory_variant in territoire_lower or territoire_lower in territory_variant:
                return standard_name

        # Sinon, retourner titre propre
        return territoire.strip().title()

    def _validate_loaded_data(self, data):
        """Valide et nettoie les données chargées."""
        valid_data = []
        invalid_count = 0

        for item in data:
            if not item.get('question', '').strip():
                invalid_count += 1
                continue

            if not item.get('answer', '').strip():
                invalid_count += 1
                continue

            item['question'] = self._clean_text(item['question'])
            item['answer'] = self._clean_text(item['answer'])

            # Marquer qualité
            item['data_quality'] = 'good'
            valid_data.append(item)

        if invalid_count > 0:
            print(f"⚠️ {invalid_count} entrées invalides supprimées")

        print(f"✓ {len(valid_data)} entrées validées avec succès")
        return valid_data

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = re.sub(r'\s+([.!?])', r'\1', text)
        return text

    def _extract_territory_from_text(self, question: str, response: str) -> str:
        text_to_search = f"{question} {response}".lower()

        for territory_variant, standard_name in self.morocco_territories.items():
            if territory_variant in text_to_search:
                return standard_name

        territory_patterns = [
            (r'ensemble du territoire national', 'Ensemble du territoire national'),
            (r'territoire national', 'Ensemble du territoire national'),
            (r'maroc', 'Ensemble du territoire national'),
            (r'royaume', 'Ensemble du territoire national'),
        ]

        for pattern, territory in territory_patterns:
            if re.search(pattern, text_to_search, re.IGNORECASE):
                return territory

        return 'Territoire non spécifié'

    def _classify_question_from_text(self, question: str, response: str) -> str:
        text = f"{question} {response}".lower()

        if 'population légale' in text or 'population legale' in text:
            return 'population_legale'
        elif 'population municipale' in text:
            return 'population_municipale'
        elif any(term in text for term in ["nombre d'habitants", "combien d'habitants", 'habitants', 'habitante']):
            return 'population_count'
        elif re.search(r'\d+-\d+\s+ans', text) or re.search(r'tranche.*?âge', text):
            return 'demographics_age'
        elif any(term in text for term in ['sexe', 'masculin', 'féminin', 'genre']):
            return 'demographics_gender'
        elif any(term in text for term in ['emploi', 'chômage', 'activité', 'travail']):
            return 'employment'
        elif any(term in text for term in ['éducation', 'scolarisation', 'école', 'scolarité']):
            return 'education'
        elif any(term in text for term in ['mariage', 'matrimonial', 'célibataire', 'marié']):
            return 'marital_status'
        elif any(term in text for term in ['logement', 'habitat', 'ménage', 'menage']):
            return 'housing'
        else:
            return 'demographic_general'

    def _extract_variable_from_text(self, question: str, response: str) -> str:
        text = f"{question} {response}".lower()

        hcp_variables = [
            ('population légale', 'population_legale'),
            ('population legale', 'population_legale'),
            ('population municipale', 'population_municipale'),
            ("nombre d'habitants", 'population_count'),
            ('habitants', 'population_count'),
            (r'(\d+-\d+)\s+ans', 'tranche_age'),
            ('pourcentage', 'percentage'),
            ('pourcentage.*âge', 'repartition_age'),
            ('masculin', 'pourcentage_masculin'),
            ('féminin', 'pourcentage_feminin'),
            ('célibataire', 'matrimonial_celibataire'),
        ]

        for pattern, variable in hcp_variables:
            if re.search(pattern, text):
                return variable

        return 'indicateur_demographique'

    def _extract_gender_from_text(self, question: str, response: str) -> str:
        text = f"{question} {response}".lower()

        if 'masculin' in text:
            return 'masculin'
        elif 'féminin' in text or 'feminin' in text:
            return 'feminin'
        elif any(term in text for term in ['ensemble', 'total', "pour l'ensemble", 'tous']):
            return 'ensemble'
        else:
            return 'ensemble'

    def _extract_numerical_indicators_from_text(self, response: str):
        """Extraction améliorée des nombres et pourcentages.

        Retourne un dict contenant au minimum:
          - population (int) si détecté
          - percentage (float) si détecté
          - raw_numbers (list) : tous les nombres extraits
          - indicator_type
        """
        indicators = {}
        text = response or ''

        # Extraire tous les nombres (fr: spaces as thousands, en: commas). Accepte séparateurs non-alphanum.
        num_pattern = r"(\d{1,3}(?:[\s\u00A0]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?)"
        num_matches = re.findall(num_pattern, text)

        raw_numbers = []
        for n in num_matches:
            cleaned = n.replace('\u00A0', ' ').replace(' ', '').replace(',', '.')
            try:
                if '.' in cleaned:
                    val = float(cleaned)
                else:
                    val = int(cleaned)
                raw_numbers.append(val)
            except Exception:
                continue

        indicators['raw_numbers'] = raw_numbers

        # Population heuristics : si le mot 'habitants' présent, prendre le premier entier élevé
        if re.search(r'habitant', text, re.IGNORECASE) and raw_numbers:
            # prendre le premier entier >= 1000 ou le premier entier
            pop_candidate = next((x for x in raw_numbers if isinstance(x, int) and x >= 1000), raw_numbers[0])
            try:
                indicators['population'] = int(pop_candidate)
            except Exception:
                pass

        # Pourcentage
        pct_pattern = r"(\d+(?:[.,]\d+)?)\s*%"
        pct_matches = re.findall(pct_pattern, text)
        if pct_matches:
            try:
                pct = float(pct_matches[0].replace(',', '.'))
                indicators['percentage'] = pct
            except Exception:
                pass

        # Déterminer indicator_type
        if 'population' in indicators:
            indicators['indicator_type'] = 'population'
        elif 'percentage' in indicators:
            indicators['indicator_type'] = 'percentage'
        elif raw_numbers:
            indicators['indicator_type'] = 'count'
        else:
            indicators['indicator_type'] = 'unknown'

        return indicators

    def create_embeddings(self, qa_pairs):
        """Crée des embeddings (np.array float32) et construit un index FAISS.

        Attendu qa_pairs: list of dicts avec au moins 'question' (ou 'input_text').
        """
        if not qa_pairs:
            print("Aucune paire QA disponible pour créer des embeddings")
            return np.array([])

        enhanced_questions = []
        for pair in qa_pairs:
            question = pair.get('question') or pair.get('input_text') or ''
            territory = pair.get('territory') or pair.get('territory', '')
            indicateur = pair.get('variable') or pair.get('indicator') or ''

            context_parts = []
            if territory and territory != 'Territoire non spécifié':
                context_parts.append(territory)
            if indicateur and indicateur != 'indicateur_demographique':
                context_parts.append(indicateur)

            if context_parts:
                enhanced_question = f"{' '.join(context_parts)}: {question}"
            else:
                enhanced_question = question

            enhanced_questions.append(enhanced_question)

        # Calcul embeddings
        embeddings = self.embedding_model.encode(enhanced_questions, convert_to_tensor=False)
        embeddings = np.asarray(embeddings, dtype='float32')
        self.embeddings = embeddings

        # FAISS Index
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)

        # Normaliser puis ajouter
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        print(f"Index FAISS créé avec {len(embeddings)} embeddings de dimension {dimension}")
        return embeddings

    def find_similar_questions(self, query: str, k: int = 3):
        if self.index is None:
            return []

        processed_query = self._preprocess_query(query)
        query_embedding = self.embedding_model.encode([processed_query], convert_to_tensor=False)
        query_embedding = np.asarray(query_embedding, dtype='float32')
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)

        results = []
        threshold = getattr(self.config, 'SIMILARITY_THRESHOLD', 0.3)

        for idx, score in zip(indices[0], scores[0]):
            if float(score) > float(threshold):
                results.append((int(idx), float(score)))

        return results

    def _preprocess_query(self, query: str) -> str:
        query = self._clean_text(query)
        query_lower = query.lower()
        for territory_variant, standard_name in self.morocco_territories.items():
            if territory_variant in query_lower:
                query = re.sub(re.escape(territory_variant), standard_name, query, flags=re.IGNORECASE)
                break
        return query

    def create_qa_pairs(self):
        if self.combined_data:
            qa_pairs = self.combined_data.copy()
        elif self.data is not None and not self.data.empty:
            qa_pairs = self.data.to_dict('records')
        else:
            return []

        for pair in qa_pairs:
            pair['input_text'] = pair.get('question')
            pair['target_text'] = pair.get('answer')

            context_parts = []
            if pair.get('territory'):
                context_parts.append(f"territory:{pair['territory']}")
            if pair.get('question_type'):
                context_parts.append(f"type:{pair['question_type']}")
            if pair.get('sexe'):
                context_parts.append(f"sexe:{pair['sexe']}")
            if pair.get('variable'):
                context_parts.append(f"indicator:{pair['variable']}")
            if pair.get('source_data'):
                context_parts.append(f"source:{pair['source_data']}")

            pair['context'] = ','.join(context_parts)

        print(f"Nombre de paires QA extraites: {len(qa_pairs)}")
        return qa_pairs

    def get_statistics(self):
        if not self.combined_data:
            return {"message": "Aucune donnée chargée"}

        stats = {
            "total_qa_pairs": len(self.combined_data),
            "territories": {},
            "question_types": {},
            "indicators": {},
            "sources": {},
            "gender_distribution": {},
            "data_quality": {"good": 0, "suspect": 0},
            "data_sources": {}
        }

        for item in self.combined_data:
            territory = item.get('territory', 'Non spécifié')
            stats["territories"][territory] = stats["territories"].get(territory, 0) + 1

            q_type = item.get('question_type', 'Non classé')
            stats["question_types"][q_type] = stats["question_types"].get(q_type, 0) + 1

            indicator = item.get('variable', 'Non spécifié')
            stats["indicators"][indicator] = stats["indicators"].get(indicator, 0) + 1

            source = item.get('source_data', 'Non spécifié')
            stats["sources"][source] = stats["sources"].get(source, 0) + 1

            gender = item.get('sexe', 'Non spécifié')
            stats["gender_distribution"][gender] = stats["gender_distribution"].get(gender, 0) + 1

            quality = item.get('data_quality', 'unknown')
            if quality in stats["data_quality"]:
                stats["data_quality"][quality] += 1

            data_source = item.get('data_source', 'unknown')
            stats["data_sources"][data_source] = stats["data_sources"].get(data_source, 0) + 1

        return stats

    def analyze_data_coverage(self):
        if not self.combined_data:
            print("Aucune donnée à analyser")
            return

        print("\n=== ANALYSE DE COUVERTURE DES DONNÉES (NOUVELLE STRUCTURE) ===")

        territory_stats = {}
        for item in self.combined_data:
            territory = item.get('territory', 'Non spécifié')
            if territory not in territory_stats:
                territory_stats[territory] = {'count': 0, 'indicators': set(), 'types': set()}
            territory_stats[territory]['count'] += 1
            territory_stats[territory]['indicators'].add(item.get('variable', 'unknown'))
            territory_stats[territory]['types'].add(item.get('question_type', 'unknown'))

        print(f"\n📍 COUVERTURE PAR TERRITOIRE ({len(territory_stats)} territoires):")
        sorted_territories = sorted(territory_stats.items(), key=lambda x: x[1]['count'], reverse=True)
        for territory, stats in sorted_territories[:10]:
            display_territory = territory[:50] + "..." if len(territory) > 50 else territory
            print(f"  {display_territory}: {stats['count']} questions, {len(stats['indicators'])} indicateurs")

        indicator_stats = {}
        for item in self.combined_data:
            indicator = item.get('variable', 'unknown')
            if indicator not in indicator_stats:
                indicator_stats[indicator] = {'count': 0, 'territories': set()}
            indicator_stats[indicator]['count'] += 1
            indicator_stats[indicator]['territories'].add(item.get('territory', 'Non spécifié'))

        print(f"\n📊 COUVERTURE PAR INDICATEUR ({len(indicator_stats)} indicateurs):")
        sorted_indicators = sorted(indicator_stats.items(), key=lambda x: x[1]['count'], reverse=True)
        for indicator, stats in sorted_indicators[:10]:
            print(f"  {indicator}: {stats['count']} questions, {len(stats['territories'])} territoires")

        source_stats = {}
        for item in self.combined_data:
            source = item.get('source_data', 'unknown')
            source_stats[source] = source_stats.get(source, 0) + 1

        print(f"\n🔄 COUVERTURE PAR SOURCE DE DONNÉES:")
        for source, count in sorted(source_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.combined_data)) * 100
            print(f"  {source}: {count} ({percentage:.1f}%)")

        print(f"\n✅ TOTAL: {len(self.combined_data)} paires question-réponse validées")

    def export_data(self, filepath: str, format: str = 'json'):
        try:
            if format.lower() == 'json':
                export_data = {
                    "metadata": {
                        "export_date": pd.Timestamp.now().isoformat(),
                        "total_qa_pairs": len(self.combined_data),
                        "export_format": "hcp_chatbot_format"
                    },
                    "qa_pairs": self.combined_data
                }
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
            elif format.lower() == 'csv':
                if self.data is not None:
                    export_data = []
                    for item in self.combined_data:
                        flat_item = item.copy()
                        if isinstance(flat_item.get('indicators'), dict):
                            flat_item.update({f"indicator_{k}": v for k, v in flat_item['indicators'].items()})
                            del flat_item['indicators']
                        export_data.append(flat_item)

                    pd.DataFrame(export_data).to_csv(filepath, index=False, encoding='utf-8')
                else:
                    return False
            else:
                print(f"Format {format} non supporté")
                return False

            print(f"Données exportées vers {filepath} (format: {format})")
            return True

        except Exception as e:
            print(f"Erreur lors de l'export: {e}")
            return False

    def search_qa_pairs(self, query: str, filters: dict = None):
        if not self.combined_data:
            return []

        results = []
        query_lower = query.lower()

        for item in self.combined_data:
            if (query_lower in item['question'].lower() or
                    query_lower in item['answer'].lower() or
                    query_lower in item.get('variable', '').lower()):

                if filters:
                    match = True
                    for key, value in filters.items():
                        item_value = item.get(key, '')
                        if isinstance(value, list):
                            if item_value not in value:
                                match = False
                                break
                        else:
                            if item_value != value:
                                match = False
                                break
                    if not match:
                        continue

                results.append(item)

        return results

    def get_sample_data(self, n: int = 5):
        if not self.combined_data:
            return []

        import random
        sample_size = min(n, len(self.combined_data))
        return random.sample(self.combined_data, sample_size)


# Petit utilitaire de test
def test_new_structure_processor(config):
    processor = HCPDataProcessor(config)

    print("=== TEST DU PROCESSEUR POUR NOUVELLE STRUCTURE HCP ===\n")

    data = processor.load_all_data()

    if data.empty:
        print("❌ Aucune donnée chargée")
        return None

    qa_pairs = processor.create_qa_pairs()
    print(f"\n✅ {len(qa_pairs)} paires QA créées avec succès")

    print("\n📋 ÉCHANTILLON DES DONNÉES:")
    sample = processor.get_sample_data(3)
    for i, item in enumerate(sample, 1):
        print(f"\n{i}. Question: {item['question'][:120]}...")
        print(f"   Réponse: {item['answer'][:120]}...")
        print(f"   Territoire: {item.get('territory', 'N/A')}")
        print(f"   Indicateur: {item.get('variable', 'N/A')}")
        print(f"   Genre: {item.get('sexe', 'N/A')}")
        print(f"   Source: {item.get('source_data', 'N/A')}")

    stats = processor.get_statistics()
    print(f"\n📊 STATISTIQUES DÉTAILLÉES:")
    print(f"  - Total paires QA: {stats['total_qa_pairs']}")
    print(f"  - Territoires uniques: {len(stats['territories'])}")
    print(f"  - Types de questions: {len(stats['question_types'])}")
    print(f"  - Indicateurs uniques: {len(stats['indicators'])}")

    processor.analyze_data_coverage()

    if qa_pairs:
        print("\n=== TEST DE RECHERCHE SÉMANTIQUE ===")
        embeddings = processor.create_embeddings(qa_pairs)

        test_queries = [
            "Quelle est la population légale du Maroc ?",
            "Combien d'habitants au niveau national ?",
            "Pourcentage population masculine ?",
            "Population féminine territoire national"
        ]

        for query in test_queries:
            results = processor.find_similar_questions(query, k=2)
            print(f"\n🔍 Requête: {query}")
            if results:
                for idx, score in results:
                    if idx < len(qa_pairs):
                        qa_item = qa_pairs[idx]
                        print(f"   Similarité {score:.3f}: {qa_item['question'][:60]}...")
                        print(f"   → {qa_item['answer'][:60]}...")
            else:
                print("   Aucun résultat trouvé")

    print(f"\n✅ Test terminé avec succès!")
    return processor


if __name__ == "__main__":
    try:
        from config import Config
        test_new_structure_processor(Config)
    except ImportError:
        print("Impossible d'importer Config. Assurez-vous que config.py existe.")
