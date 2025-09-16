# 🇲🇦 HCP Chatbot — Assistant IA pour les données démographiques d'Agadir

Chatbot intelligent spécialisé dans l'analyse et la consultation des données officielles du **Haut-Commissariat au Plan (HCP)** pour la **région d'Agadir**. Accès instantané aux statistiques démographiques locales couvrant les communes, provinces et territoires de la région d'Agadir.

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://hub.docker.com/r/bouachrineyassine/hcp-chatbot)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green?logo=flask)](https://flask.palletsprojects.com/)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow?logo=python)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-Latest-orange)](https://huggingface.co/transformers/)

## 📊 À propos du projet

Le **HCP Chatbot** facilite l'accès aux données démographiques officielles **de la région d'Agadir** en utilisant l'IA pour répondre aux questions sur :

* **Population légale et municipale** au niveau de la région, des provinces et des communes d'Agadir
* **Indicateurs démographiques** (répartition par âge, genre, état matrimonial)
* **Statistiques socio-économiques** locales (emploi, éducation, logement)
* **Répartition territoriale détaillée** au sein de la région (zones urbaines/rurales, découpages administratifs)

### 🎯 Fonctionnalités

- ✅ **Recherche sémantique** 
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


## 🧪 Tests

### API Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Interface web |


### Réponse API

```json
  {
    "timestamp": "2025-09-11 20:29:54",
    "query": "Pour l'ensemble de la population, quel est le nombre de population légale à Région de Souss-Massa ?",
    "query_corrected": "Pour ensemble de la population, quel est le nombre de population légale à Région de Souss-Massa ?",
    "response": "Pour l'ensemble de la population, le nombre de population légale à Région de Souss-Massa est de 3020431 personnes.",
    "territory_detected": null,
    "indicator_detected": "population_legale",
    "source_detected": "population",
    "response_length": 114,
    "model_used": "search_only"
  },
```

## 📊 Données Supportées

### Couverture Géographique
- 🗺️ **Régions** : La région d'agadir
- 🏢 **Provinces** : Tout les provinces de la région d'agadir
- 🏘️ **Communes** : Tout les communes de la région d'agadir

### Indicateurs
- **👥 Population** : Légale/municipale, par genre, par âge
- **💒 État Matrimonial** : Célibataires, mariés, divorcés, veufs
- **🎂 Tranches d'Âge** : 0-4, 5-9, ..., 85+
- **💼 Emploi** : Population active, chômage
- **🎓 Éducation** : Scolarisation, alphabétisation
- ....


## 📞 Support

- 📧 **Contact** : bouachrinyassin0@gmail.com

## 📜 Licence

Ce projet est sous licence **MIT**. 

### Sources

- 🏛️ **Haut-Commissariat au Plan (HCP)** pour les données officielles
- 🤗 **Hugging Face** pour DistilGPT2 et les transformers
- 🐳 **Docker** pour la containerisation

---

<div align="center">

**🇲🇦 Pour faciliter l'accès aux données démographiques de la région d'agadir**

[⭐ Star ce projet](https://github.com/yassinebouachrine/hcp-chatbot) • [🐳 Docker Hub](https://hub.docker.com/r/bouachrineyassine/hcp-chatbot)

</div>
