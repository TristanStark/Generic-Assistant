# Configuration
IMAGE_NAME = rag-api
TAG = latest
ARCHIVE_NAME = $(IMAGE_NAME).tar.gz
REMOTE_USER = ubuntu
REMOTE_HOST = 5.196.27.249
REMOTE_PATH = /home/ubuntu

# =============================================
# 🔨 Étape 1 : Build de l'image Docker
# =============================================
build:
	docker build -t $(IMAGE_NAME):$(TAG) .

# =============================================
# 📦 Étape 2 : Export et compression
# =============================================
export:
	docker save $(IMAGE_NAME):$(TAG) | gzip > $(ARCHIVE_NAME)

# =============================================
# 🚀 Étape 3 : Envoi vers le VPS
# =============================================
scp:
	scp $(ARCHIVE_NAME) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_PATH)/

# =============================================
# ⚡ Étape 4 : Déploiement complet
# =============================================
deploy: build export scp
	@echo "[REUSSITE] Image transférée vers le VPS : $(REMOTE_HOST)"

# =============================================
# 📥 (sur le VPS) Charger l'image transférée
# =============================================
load:
	gunzip -f $(ARCHIVE_NAME)
	docker load < $(basename $(ARCHIVE_NAME))

# =============================================
# 🧹 Nettoyage local
# =============================================
clean:
	rm -f $(ARCHIVE_NAME)
