# Étape 1 : Base Python légère
FROM python:3.12-slim

# Étape 2 : Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/tmp \
    TORCH_HOME=/tmp

# Étape 3 : Install packages système requis
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Étape 4 : Création du dossier de travail
WORKDIR /app

# Étape 5 : Copier uniquement les fichiers utiles
COPY requirements.txt .

# Étape 6 : Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip

# Étape 7 : Copier le reste du code
COPY . .

# Étape 8 : Exposer le port Flask
EXPOSE 5000

# Étape 9 : Commande de lancement
CMD ["python", "app.py"]
