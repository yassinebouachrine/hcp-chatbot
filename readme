# 🇲🇦 HCP Chatbot - Assistant IA pour Données Démographiques du Maroc

> Chatbot intelligent spécialisé dans l'analyse et la consultation des données officielles du **Haut-Commissariat au Plan (HCP)** du Maroc. Accès instantané à plus de 140 000 statistiques démographiques couvrant 2000+ territoires marocains.

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://hub.docker.com/r/bouachrineyassine/hcp-chatbot)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green?logo=flask)](https://flask.palletsprojects.com/)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow?logo=python)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-Latest-orange)](https://huggingface.co/transformers/)

## 📊 À propos du projet

Le **HCP Chatbot** est un assistant conversationnel intelligent développé pour faciliter l'accès aux données démographiques officielles du Maroc. Il utilise des techniques avancées d'IA pour comprendre et répondre aux questions sur :

- **Population légale et municipale** par région, province, commune
- **Indicateurs démographiques** (âge, genre, état matrimonial)
- **Statistiques socio-économiques** (emploi, éducation, logement)
- **Données des ménages** et structures familiales
- **Répartition territoriale** détaillée du Royaume

### 🎯 Fonctionnalités principales

- ✅ **Recherche sémantique avancée** dans 140 000+ statistiques HCP
- ✅ **Interface web intuitive** avec chat en temps réel
- ✅ **API REST complète** pour intégrations tierces
- ✅ **Support multilingue** (Français/Arabe)
- ✅ **Réponses contextuelles** basées sur l'IA
- ✅ **Déploiement Docker** prêt pour production

## 🏗️ Architecture et Technologies

### Stack Technique

| Composant | Technologie | Version | Description |
|-----------|-------------|---------|-------------|
| **Backend** | Python | 3.9+ | Logique métier et API |
| **Framework Web** | Flask | 2.3.3 | Serveur web et endpoints |
| **Modèle Base** | DistilGPT2 | Hugging Face | Génération de texte optimisée |
| **Embeddings** | all-MiniLM-L6-v2 | Sentence-Transformers | Recherche sémantique rapide |
| **IA/NLP** | Transformers | Latest | Pipeline de traitement |
| **Frontend** | HTML/CSS/JS | - | Interface utilisateur |
| **Base de données** | JSON | - | Stockage des données HCP |
| **Conteneurisation** | Docker | Latest | Déploiement unifié |

### Structure du Projet

```
HCP-CHATBOT/
├── 📁 src/                    # Code source principal
│   ├── 🐍 chatbot.py          # Logique du chatbot IA
│   ├── 🔍 data_processor.py   # Traitement données HCP
│   └── 🤖 model_trainer.py    # Entraînement modèles
├── 📁 data/                   # Données HCP
│   ├── 📊 indicators.json     # 140K+ statistiques (200MB+)
│   └── 📋 conversation_history.json
├── 📁 models/                 # Modèles IA pré-entraînés
│   └── 📁 models_hcp/         # Modèles spécialisés HCP
├── 📁 templates/              # Interface web
│   └── 🌐 index.html          # Page principale
├── 📁 static/                 # Ressources statiques
│   ├── 🎨 css/               # Styles
│   ├── ⚡ js/                # JavaScript
│   └── 🖼️ images/            # Images/logos
├── 🐳 docker-compose.yml      # Orchestration Docker
├── 🐳 Dockerfile             # Image Docker
├── 🔧 config.py              # Configuration système
├── 🚀 app.py                 # Application Flask principale
└── 📋 requirements.txt       # Dépendances Python
```

### 🧠 Intelligence Artificielle

Le chatbot utilise une architecture hybride optimisée pour les données HCP :

#### **Modèle de Génération : DistilGPT2**
- **Avantages** : Version distillée de GPT-2, plus rapide et légère
- **Utilisation** : Génération de réponses contextuelles fluides
- **Performance** : 82M paramètres vs 124M pour GPT-2 standard
- **Spécialisation** : Fine-tuné sur les données démographiques HCP

#### **Modèle d'Embeddings : all-MiniLM-L6-v2**
- **Architecture** : 22M paramètres, très efficace
- **Vitesse** : ~14 000 phrases/seconde sur CPU
- **Qualité** : Score SBERT de 82.05 sur tâches sémantiques
- **Langues** : Support multilingue (Français inclus)

#### **Pipeline de Traitement**
1. **Encodage Sémantique** : Conversion de la question en vecteur 384D
2. **Recherche Vectorielle** : Similarité cosinus dans la base HCP
3. **Classification Contextuelle** : Détection territoire/indicateur/type
4. **Génération Guidée** : DistilGPT2 avec contexte HCP spécialisé

#### **Optimisations Spécifiques HCP**
- **Index Vectoriel** : 140K+ embeddings pré-calculés et mis en cache
- **Filtrage Intelligent** : Priorisation des données les plus pertinentes
- **Adaptation Terminologique** : Vocabulaire spécialisé démographie Maroc

## 🚀 Installation et Utilisation

### Option 1: Docker Hub (Recommandé)

Le moyen le plus simple pour démarrer :

```bash
# 1. Télécharger l'image depuis Docker Hub
docker pull bouachrineyassine/hcp-chatbot:latest

# 2. Lancer le chatbot
docker run -d \
  --name hcp-chatbot \
  -p 5000:5000 \
  -e FLASK_ENV=production \
  bouachrineyassine/hcp-chatbot:latest

# 3. Accéder à l'interface
# Ouvrir: http://localhost:5000
```

### Option 2: Docker Compose (Développement)

Pour un contrôle total et des personnalisations :

```bash
# 1. Cloner le repository
git clone https://github.com/yassinebouachrine/hcp-chatbot.git
cd hcp-chatbot

# 2. Configuration (optionnel)
cp .env.template .env
# Éditer .env selon vos besoins

# 3. Construire et lancer
docker-compose up -d

# 4. Vérifier les logs
docker-compose logs -f

# 5. Accéder à l'application
# Interface: http://localhost:5000
# API: http://localhost:5000/chat
```

### Option 3: Installation Locale (Développeurs)

Pour développement et personnalisations avancées :

```bash
# 1. Cloner et préparer
git clone https://github.com/yassinebouachrine/hcp-chatbot.git
cd hcp-chatbot

# 2. Environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les données HCP
# S'assurer que data/indicators.json existe et contient les données

# 5. Lancer l'application
python app.py

# Application disponible sur http://localhost:5000
```

## 🔧 Configuration Avancée

### Configuration Avancée Modèles

```env
# Configuration Flask
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=False

# Configuration Modèles IA
MODEL_PATH=models/models_hcp        # Chemin modèle fine-tuné
BASE_MODEL=distilgpt2              # Modèle base Hugging Face
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Embeddings

# Configuration Génération
SIMILARITY_THRESHOLD=0.75          # Seuil de pertinence (0.0-1.0)
MAX_LENGTH=128                     # Longueur max des réponses
TEMPERATURE=0.3                    # Créativité des réponses (0.1-1.0)
TOP_K=50                          # Top-K sampling
TOP_P=0.95                        # Nucleus sampling

# Configuration Performance  
BATCH_SIZE=64                     # Taille de batch (ajuster selon RAM)
EMBEDDING_BATCH_SIZE=32           # Batch pour embeddings
CACHE_EMBEDDINGS=True             # Cache des embeddings calculés

# Configuration HuggingFace
HUGGING_FACE_TOKEN=your_token_here # Token pour modèles privés

# Configuration GPU (si disponible)
USE_CUDA=False                    # True pour utiliser GPU
FP16=False                        # Précision mixte pour GPU
DEVICE_MAP=auto                   # Répartition automatique GPU

# Logging et Debug
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR
SAVE_CONVERSATIONS=True           # Historique des conversations
```

### 🎛️ Optimisation Performances

**Configuration par Type de Machine :**

#### Machine Standard (4-8GB RAM)
```env
# Optimisé pour ressources limitées
BASE_MODEL=distilgpt2
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
BATCH_SIZE=16
EMBEDDING_BATCH_SIZE=8
MAX_LENGTH=96
USE_CUDA=False
FP16=False
CACHE_EMBEDDINGS=True
```

#### Machine Puissante (16GB+ RAM)
```env
# Optimisé pour performance maximale
BASE_MODEL=distilgpt2
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
BATCH_SIZE=64
EMBEDDING_BATCH_SIZE=32
MAX_LENGTH=128
USE_CUDA=True  # Si GPU disponible
FP16=True
CACHE_EMBEDDINGS=True
```

#### Serveur Production (32GB+ RAM)
```env
# Configuration serveur haute charge
BASE_MODEL=distilgpt2
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
BATCH_SIZE=128
EMBEDDING_BATCH_SIZE=64
MAX_LENGTH=150
USE_CUDA=True
FP16=True
CACHE_EMBEDDINGS=True
WORKERS=4  # Processus Flask multiples
```

#### **Pourquoi ces Modèles ?**

| Modèle | Avantage | Taille | Performance |
|--------|----------|--------|-------------|
| **DistilGPT2** | 40% plus rapide que GPT2 | 82M params | Génération fluide |
| **all-MiniLM-L6-v2** | Le plus rapide des SBERT | 22M params | 14K phrases/sec |

## 🧪 Test et Validation

### Endpoints API Disponibles

| Endpoint | Méthode | Description | Exemple |
|----------|---------|-------------|---------|
| `/` | GET | Interface web principale | - |
| `/health` | GET | Statut système | `{"status": "healthy"}` |
| `/chat` | POST | Chat avec IA | `{"message": "Population Casablanca"}` |
| `/territories` | GET | Liste territoires HCP | Tous les territoires disponibles |

### Exemples de Test API

```bash
# Test basique - Population nationale
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Quelle est la population du Maroc?"}'

# Test territorial - Ville spécifique  
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Population de Rabat"}'

# Test indicateur démographique
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Pourcentage de femmes mariées au niveau national"}'

# Test âge spécifique
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Population de 25-29 ans à Casablanca"}'
```

### Réponse API Type

```json
{
  "response": "La population légale du Maroc (Ensemble du territoire national) est de 33 848 242 habitants selon les dernières données HCP.",
  "status": "success",
  "metadata": {
    "model_used": "semantic_search",
    "qa_pairs_count": 140567,
    "territory_detected": "Ensemble du territoire national",
    "indicator_detected": "population_legale",
    "confidence_score": 0.89,
    "response_time_ms": 245
  }
}
```

## 📊 Données Supportées

### Couverture Géographique

- **🏛️ Niveau National** : Ensemble du territoire
- **🗺️ Régions** : 12 régions administratives
- **🏢 Provinces** : 75+ provinces et préfectures  
- **🏘️ Communes** : 1500+ communes urbaines et rurales
- **📍 Autres Territoires** : Arrondissements, districts

### Indicateurs Démographiques

| Catégorie | Indicateurs | Exemples |
|-----------|-------------|----------|
| **👥 Population** | Population légale/municipale | Total, par genre, par âge |
| **💒 État Matrimonial** | Célibataires, mariés, divorcés, veufs | Par genre, par âge |
| **🎂 Tranches d'Âge** | 0-4, 5-9, ..., 85+ | Population détaillée |
| **💼 Emploi** | Population active, chômage | Taux d'activité |
| **🎓 Éducation** | Scolarisation, alphabétisation | Par niveau |
| **🏠 Ménages** | Taille moyenne, composition | Structure familiale |

### Questions Types Supportées

- ❓ **Population** : "Population de [Territoire]"
- ❓ **Genre** : "Nombre de femmes à [Lieu]"
- ❓ **Âge** : "Population de 25-29 ans au Maroc"
- ❓ **Matrimonial** : "Pourcentage de mariés à [Ville]"
- ❓ **Comparaison** : "Différence population entre Rabat et Casablanca"
- ❓ **Évolution** : "Évolution population [Région]"

## 🔍 Monitoring et Maintenance

### Surveillance Système

```bash
# Vérifier l'état de santé
curl http://localhost:5000/health

# Logs en temps réel
docker-compose logs -f hcp-chatbot

# Utilisation ressources
docker stats hcp-chatbot-app

# Informations détaillées
docker-compose exec hcp-chatbot htop
```

### Mise à jour des Données

```bash
# Nouvelle version des données HCP
cp nouvelles_donnees.json data/indicators.json

# Redémarrer pour appliquer
docker-compose restart hcp-chatbot

# Vérifier le chargement
curl http://localhost:5000/health
```

## 🛡️ Production et Sécurité

### Déploiement Production

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  hcp-chatbot:
    image: bouachrineyassine/hcp-chatbot:latest
    restart: always
    environment:
      - FLASK_ENV=production
      - FLASK_DEBUG=False
      - LOG_LEVEL=WARNING
    ports:
      - "80:5000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Configuration Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name votre-domaine.com;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 🤝 Contribution et Développement

### Structure de Développement

```bash
# Fork du repository
git clone https://github.com/votre-username/hcp-chatbot.git

# Branche de développement
git checkout -b feature/nouvelle-fonctionnalite

# Environnement de dev
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Outils de dev

# Tests
python -m pytest tests/

# Lancer en mode développement
export FLASK_DEBUG=True
python app.py
```

### Roadmap Futur

- 🔄 **API GraphQL** pour requêtes complexes
- 📱 **Application Mobile** React Native
- 🗣️ **Support Vocal** avec reconnaissance/synthèse
- 📈 **Visualisations Interactives** avec charts dynamiques
- 🔍 **Recherche Avancée** avec filtres multiples
- 🌐 **Multilingue Complet** (Français, Arabe, Berbère)
- 🔗 **Intégrations** avec autres sources officielles

## 📞 Support et Contact

### Ressources Utiles

- 📖 **Documentation** : Disponible dans `/docs`
- 🐛 **Issues** : [GitHub Issues](https://github.com/yassinebouachrine/hcp-chatbot/issues)
- 💬 **Discussions** : [GitHub Discussions](https://github.com/yassinebouachrine/hcp-chatbot/discussions)
- 📧 **Contact** : [yassine.bouachrine@example.com](mailto:yassine.bouachrine@example.com)

### Diagnostic Rapide

```bash
# Vérification complète des modèles
docker-compose exec hcp-chatbot python -c "
from src.chatbot import HCPChatbotAdapted
from config import Config
import torch
print('✅ Modules OK')

# Vérifier les modèles
print(f'📊 Modèle base: {Config.BASE_MODEL}')
print(f'🔍 Modèle embeddings: {Config.EMBEDDING_MODEL}')
print(f'💾 Chemin modèles: {Config.MODEL_PATH}')
print(f'🚀 CUDA disponible: {torch.cuda.is_available()}')

chatbot = HCPChatbotAdapted(Config)
print('✅ Chatbot initialisé avec DistilGPT2 + MiniLM')

response = chatbot.get_response('Population du Maroc')
print(f'✅ Test réponse: {response[:50]}...')
print(f'📈 Embeddings chargés: {len(chatbot.embeddings) if hasattr(chatbot, \"embeddings\") else \"N/A\"}')
"
```

## 📜 Licence et Remerciements

### Licence

Ce projet est sous licence **MIT**. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

### Remerciements

- 🏛️ **Haut-Commissariat au Plan (HCP)** pour les données officielles
- 🤗 **Hugging Face** pour les modèles de transformers
- 🐳 **Docker** pour la containerisation
- 🌶️ **Flask** pour le framework web
- 👥 **Communauté Open Source** pour les outils et bibliothèques

---

<div align="center">

**🇲🇦 Fait avec ❤️ pour faciliter l'accès aux données démographiques du Maroc**

[⭐ Star ce projet](https://github.com/yassinebouachrine/hcp-chatbot) • [🐳 Docker Hub](https://hub.docker.com/r/bouachrineyassine/hcp-chatbot) • [📊 Démo Live](http://votre-demo.com)

</div>
