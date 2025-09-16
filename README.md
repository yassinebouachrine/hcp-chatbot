# 🇲🇦 HCP Chatbot - Assistant IA pour Données Démographiques du Maroc

Chatbot intelligent spécialisé dans l'analyse et la consultation des données officielles du **Haut-Commissariat au Plan (HCP)** du Maroc. Accès instantané à plus de 140 000 statistiques démographiques couvrant 2000+ territoires marocains.

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://hub.docker.com/r/bouachrineyassine/hcp-chatbot)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green?logo=flask)](https://flask.palletsprojects.com/)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow?logo=python)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-Latest-orange)](https://huggingface.co/transformers/)

## 📊 À propos du projet

Le **HCP Chatbot** facilite l'accès aux données démographiques officielles du Maroc en utilisant l'IA pour répondre aux questions sur :

- **Population légale et municipale** par région, province, commune
- **Indicateurs démographiques** (âge, genre, état matrimonial)  
- **Statistiques socio-économiques** (emploi, éducation, logement)
- **Répartition territoriale** détaillée du Royaume

### 🎯 Fonctionnalités

- ✅ **Recherche sémantique** dans 140 000+ statistiques HCP
- ✅ **Interface web intuitive** avec chat en temps réel
- ✅ **API REST** pour intégrations
- ✅ **Réponses contextuelles** basées sur l'IA

## 🏗️ Technologies

| Composant | Technologie | Description |
|-----------|-------------|-------------|
| **Backend** | Python 3.9+ | Logique métier et API |
| **Framework** | Flask 2.3.3 | Serveur web |
| **Modèle Base** | DistilGPT2 | Génération de texte |
| **Embeddings** | all-MiniLM-L6-v2 | Recherche sémantique |
| **Container** | Docker | Déploiement unifié |

### Structure du Projet

```
HCP-CHATBOT/
├── src/                    # Code source
│   ├── chatbot.py          # Chatbot IA
│   ├── data_processor.py   # Traitement données HCP
│   └── model_trainer.py    # Entraînement modèles
├── data/                   # Données HCP (140K+ stats)
├── models/                 # Modèles DistilGPT2 + MiniLM
├── templates/              # Interface web
├── static/                 # CSS/JS/Images
├── docker-compose.yml      # Configuration Docker
├── Dockerfile             
├── config.py              # Configuration
├── app.py                 # Application Flask
└── requirements.txt       
```

## 🚀 Installation

### Option 1: Docker Hub (Recommandé)

```bash
# Télécharger et lancer
docker pull bouachrineyassine/hcp-chatbot:latest
docker run -d --name hcp-chatbot -p 5000:5000 bouachrineyassine/hcp-chatbot:latest

# Accéder: http://localhost:5000
```

### Option 2: Docker Compose

```bash
# Cloner le projet
git clone https://github.com/yassinebouachrine/hcp-chatbot.git
cd hcp-chatbot

# Lancer
docker-compose up -d

# Interface: http://localhost:5000
```

### Option 3: Installation Locale

```bash
# Cloner et installer
git clone https://github.com/yassinebouachrine/hcp-chatbot.git
cd hcp-chatbot

# Environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Dépendances
pip install -r requirements.txt

# Lancer
python app.py
```

## 🔧 Configuration

Créer un fichier `.env` :

```env
# Configuration Flask
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=False

# Configuration Modèles
MODEL_PATH=models/models_hcp
BASE_MODEL=distilgpt2
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Performance
SIMILARITY_THRESHOLD=0.75
MAX_LENGTH=128
TEMPERATURE=0.3
BATCH_SIZE=64

# GPU (optionnel)
USE_CUDA=False
FP16=False
```

### Optimisation selon votre machine

**Machine Standard (4-8GB RAM):**
```env
BATCH_SIZE=16
MAX_LENGTH=96
USE_CUDA=False
```

**Machine Puissante (16GB+):**
```env
BATCH_SIZE=64
MAX_LENGTH=128
USE_CUDA=True
FP16=True
```

## 🧪 Tests

### API Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Interface web |
| `/health` | GET | Statut système |
| `/chat` | POST | Chat avec IA |
| `/territories` | GET | Liste territoires |

### Exemples de test

```bash
# Test population nationale
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Population du Maroc"}'

# Test ville spécifique
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Population de Casablanca"}'
```

### Réponse API

```json
{
  "response": "La population légale du Maroc est de 33 848 242 habitants selon les données HCP.",
  "status": "success",
  "metadata": {
    "territory_detected": "Ensemble du territoire national",
    "indicator_detected": "population_legale",
    "confidence_score": 0.89
  }
}
```

## 📊 Données Supportées

### Couverture Géographique
- 🏛️ **Niveau National** : Ensemble du territoire
- 🗺️ **Régions** : 12 régions administratives  
- 🏢 **Provinces** : 75+ provinces et préfectures
- 🏘️ **Communes** : 1500+ communes

### Indicateurs
- **👥 Population** : Légale/municipale, par genre, par âge
- **💒 État Matrimonial** : Célibataires, mariés, divorcés, veufs
- **🎂 Tranches d'Âge** : 0-4, 5-9, ..., 85+
- **💼 Emploi** : Population active, chômage
- **🎓 Éducation** : Scolarisation, alphabétisation

### Questions Supportées
- "Population de [Territoire]"
- "Nombre de femmes à [Lieu]"  
- "Population de 25-29 ans au Maroc"
- "Pourcentage de mariés à [Ville]"

## 🔍 Monitoring

```bash
# Vérifier l'état
curl http://localhost:5000/health

# Voir les logs
docker-compose logs -f

# Stats ressources
docker stats hcp-chatbot-app
```

## 🛡️ Production

### Docker Compose Production

```yaml
version: '3.8'
services:
  hcp-chatbot:
    image: bouachrineyassine/hcp-chatbot:latest
    restart: always
    environment:
      - FLASK_ENV=production
      - FLASK_DEBUG=False
    ports:
      - "80:5000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 🛠️ Diagnostic

```bash
# Test complet
docker-compose exec hcp-chatbot python -c "
from src.chatbot import HCPChatbotAdapted
from config import Config
print('✅ Configuration OK')

chatbot = HCPChatbotAdapted(Config)
print('✅ Chatbot initialisé avec DistilGPT2 + MiniLM')

response = chatbot.get_response('Population du Maroc')
print(f'✅ Test réponse: {response[:50]}...')
"
```

## 📞 Support

- 📖 **Documentation** : Disponible dans `/docs`
- 🐛 **Issues** : [GitHub Issues](https://github.com/yassinebouachrine/hcp-chatbot/issues)
- 📧 **Contact** : yassine.bouachrine@example.com

## 📜 Licence

Ce projet est sous licence **MIT**. 

### Remerciements

- 🏛️ **Haut-Commissariat au Plan (HCP)** pour les données officielles
- 🤗 **Hugging Face** pour DistilGPT2 et les transformers
- 🐳 **Docker** pour la containerisation

---

<div align="center">

**🇲🇦 Fait avec ❤️ pour faciliter l'accès aux données démographiques du Maroc**

[⭐ Star ce projet](https://github.com/yassinebouachrine/hcp-chatbot) • [🐳 Docker Hub](https://hub.docker.com/r/bouachrineyassine/hcp-chatbot)

</div>
