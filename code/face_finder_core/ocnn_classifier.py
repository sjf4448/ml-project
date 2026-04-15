"""Prototype Recognizer/Classifier - missing features
- MTCNN/RetinaFace to crop image to face before running model
- Add projection head for recognizer instead of raw
- FAISS for larger models
ISSUES: query + embedding db needs to be normalized vectors
"""

import pickle

import torch

from .ocnn_model import build_model


class FaceClassifier:
    def __init__(self, model_path, db_path, label_encoder_path, device="cuda"):
        self.device = device

        # Model
        self.model = build_model(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        # Database
        self.db_embeddings, self.db_labels = self.load_database(db_path)
        self.db_embeddings = self.db_embeddings.to(device)
        self.db_labels = self.db_labels.to(device)

        # Label encoder
        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

    def load_database(self, path):
        data = torch.load(path)
        return data["embeddings"], data["labels"]

    def get_embedding(self, image):
        with torch.no_grad():
            image = image.to(self.device)
            emb = self.model(image)  # already normalized
        return emb

    def recognize(self, image, threshold=0.4):
        emb = self.get_embedding(image)

        sims = torch.matmul(emb, self.db_embeddings.T)

        best_idx = sims.argmax(dim=1)
        confidence = sims.max(dim=1).values.item()
        label = self.db_labels[best_idx].item()

        if confidence < threshold:
            return "unknown", confidence

        name = self.label_encoder.inverse_transform([label])[0]
        return name, confidence
