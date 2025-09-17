# encoders/clip_embeddings.py
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Désactiver les optimisations OneDNN pour éviter les problèmes de performance

class CLIPEmbeddings:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)

    def _extract_filepath(self, text):
        """
        Essaie d'extraire le chemin du fichier depuis la colonne SQL sous forme de string.
        Exemples possibles :
            "('id', './data/images/sunset_image.jpg', '2025-07-09T19:00:00')"
            "./data/images/sunset_image.jpg"
        """
        import ast

        try:
            # Essayer de parser si c'est un tuple SQL
            parsed = ast.literal_eval(text)
            if isinstance(parsed, tuple):
                for item in parsed:
                    if isinstance(item, str) and os.path.exists(item):
                        return item
        except Exception:
            pass

        # Sinon, on retourne directement si c'est un chemin
        if os.path.exists(text.strip()):
            return text.strip()

        # Dernier fallback : nettoyer les guillemets et espacer
        cleaned = text.strip().strip('"').strip("'")
        return cleaned

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            # Ici on essaye de parser le chemin
            path = self._extract_filepath(text)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found: {path}")

            image = Image.open(path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = self.model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.squeeze().cpu().numpy())
        return embeddings

    def embed_query(self, text):
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze().cpu().numpy()
