"""Prototype Recognizer/Classifier - missing features"""

import torch


class FaceClassifier:
    def __init__(self, model_path, db_path, device="cuda"):
        self.device = device
        self.model = build_model(model_path)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.db_embeddings, self.db_labels = self.load_database(db_path)

    def load_database(self, path):
        data = torch.load(path)
        return data["embeddings"], data["labels"]

    # still in progess - needs MTCNN/RetinaFace to get face before model runs on it
    def get_embedding(self, image):
        with torch.no_grad():
            emb = self.model(image.to(self.device))
            emb = torch.nn.functional.normalize(emb, dim=1)
        return emb

    def recognize(self, image, threshold=0.4):
        emb = self.get_embedding(image)

        sims = torch.matmul(emb, self.db_embeddings.T)
        best_idx = sims.argmax()

        confidence = sims[0, best_idx].item()
        label = self.db_labels[best_idx]

        if confidence < threshold:
            return "unknown", confidence

        return label, confidence
