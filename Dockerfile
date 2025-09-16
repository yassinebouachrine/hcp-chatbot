# Utiliser une image Python officielle comme base
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier requirements.txt
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code source
COPY . .

# Créer les répertoires nécessaires s'ils n'existent pas
RUN mkdir -p /app/models
RUN mkdir -p /app/data
RUN mkdir -p /app/src
RUN mkdir -p /app/static
RUN mkdir -p /app/templates

# Exposer le port sur lequel l'application s'exécute
EXPOSE 5000

# Variables d'environnement pour HCP
ENV FLASK_APP=app.py
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5000
ENV FLASK_DEBUG=False
ENV PYTHONPATH=/app
ENV HUGGING_FACE_TOKEN=""
ENV USE_CUDA=False
ENV FP16=False

# Commande pour démarrer l'application
CMD ["python", "app.py"]