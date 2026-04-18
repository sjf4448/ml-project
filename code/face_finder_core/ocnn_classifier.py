"""
recognition.py — loads the trained model + embedding DB and recognizes faces.
"""

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .ocnn_config import (
    CHECKPOINT_DIR,
    EMBEDDING_DIM,
    IMAGE_SIZE,
    MTCNN_MARGIN,
    MTCNN_MIN_FACE,
    OUTPUT_DIR,
)
from .ocnn_dataset import get_val_transforms
from .ocnn_embedding_db import load_embedding_database
from .ocnn_model import build_model

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    End-to-end face recognizer:
        raw image → MTCNN alignment → embedding → cosine nearest neighbor → identity
    """

    def __init__(
        self,
        model_path: Path,
        db_path: Path | None = None,
        label_encoder_path: Path | None = None,
        device: str | None = None,
        use_faiss: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_val_transforms()
        self.use_faiss = use_faiss

        # ── Model ─────────────────────────────────────────────────────────
        self.model = build_model(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")

        # ── MTCNN ─────────────────────────────────────────────────────────
        # Import here to avoid circular import with dataset.py
        from facenet_pytorch import MTCNN

        self.mtcnn = MTCNN(
            image_size=IMAGE_SIZE,
            margin=MTCNN_MARGIN,
            min_face_size=MTCNN_MIN_FACE,
            keep_all=False,
            post_process=False,
            device=self.device,
        )

        # ── Embedding database ─────────────────────────────────────────────
        db_path = db_path or (OUTPUT_DIR / "face_db.pt")
        label_encoder_path = label_encoder_path or (OUTPUT_DIR / "label_encoder.pkl")

        self.db_embeddings, self.db_labels = load_embedding_database(db_path)
        self.db_embeddings = self.db_embeddings.to(self.device)
        self.db_labels = self.db_labels.to(self.device)

        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        logger.info(f"Embedding DB loaded: {len(self.db_labels)} identities")

        # ── FAISS index ────────────────────────────────────────────────────
        if self.use_faiss:
            self._build_faiss_index()

    # ── FAISS ─────────────────────────────────────────────────────────────────

    def _build_faiss_index(self):
        """
        Builds an inner-product FAISS index over the DB embeddings.
        Since embeddings are L2-normalized, inner product == cosine similarity.
        """
        emb_np = self.db_embeddings.cpu().numpy().astype(np.float32)
        self.faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.faiss_index.add(emb_np)
        logger.info(f"FAISS index built: {self.faiss_index.ntotal} vectors")

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _align_face(self, pil_image: Image.Image) -> torch.Tensor | None:
        """
        Runs MTCNN on a PIL image and returns a (1, C, H, W) float tensor
        ready to pass to the model, or None if no face is detected.
        """
        face_tensor = self.mtcnn(pil_image)  # (C, H, W) or None
        if face_tensor is None:
            return None

        # Convert MTCNN output [0,255] → normalized tensor
        import numpy as np

        face_np = face_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
        augmented = self.transform(image=face_np)
        tensor = augmented["image"].unsqueeze(0)  # (1, C, H, W)
        return tensor

    def get_embedding(self, pil_image: Image.Image) -> torch.Tensor | None:
        """
        Returns a (1, EMBEDDING_DIM) L2-normalized embedding, or None
        if MTCNN finds no face in the image.
        """
        tensor = self._align_face(pil_image)
        if tensor is None:
            return None

        tensor = tensor.to(self.device)
        with torch.no_grad():
            emb = self.model(tensor)  # (1, D), already normalized
        return emb

    # ── Recognition ───────────────────────────────────────────────────────────

    def recognize(
        self,
        pil_image: Image.Image,
        threshold: float = 0.4,
    ) -> tuple[str, float]:
        """
        Recognizes the most prominent face in a PIL image.

        Returns:
            (identity_name, confidence)  — confidence is cosine similarity in [-1, 1]
            Returns ("unknown", confidence) if below threshold or no face found.
        """
        emb = self.get_embedding(pil_image)
        if emb is None:
            logger.debug("No face detected in image")
            return "unknown", 0.0

        if self.use_faiss:
            name, confidence = self._recognize_faiss(emb)
        else:
            name, confidence = self._recognize_torch(emb)

        if confidence < threshold:
            return "unknown", confidence

        return name, confidence

    def _recognize_torch(self, emb: torch.Tensor) -> tuple[str, float]:
        """Cosine similarity via matrix multiply — fine for small DBs."""
        sims = torch.matmul(emb, self.db_embeddings.T)  # (1, N)
        best_idx = sims.argmax(dim=1).item()
        confidence = sims[0, best_idx].item()
        label = self.db_labels[best_idx].item()
        name = self.label_encoder.inverse_transform([label])[0]
        return name, confidence

    def _recognize_faiss(self, emb: torch.Tensor) -> tuple[str, float]:
        """FAISS nearest-neighbor — scales to large DBs."""
        emb_np = emb.cpu().numpy().astype(np.float32)
        distances, indices = self.faiss_index.search(emb_np, k=1)
        confidence = float(distances[0, 0])
        best_idx = int(indices[0, 0])
        label = self.db_labels[best_idx].item()
        name = self.label_encoder.inverse_transform([label])[0]
        return name, confidence

    # ── Batch recognition ─────────────────────────────────────────────────────

    def recognize_batch(
        self,
        pil_images: list[Image.Image],
        threshold: float = 0.4,
    ) -> list[tuple[str, float]]:
        """Convenience wrapper for recognizing multiple images at once."""
        return [self.recognize(img, threshold) for img in pil_images]


def evaluate(
    model_path: Path | None = None,
    threshold: float = 0.4,
):
    """
    Runs recognition over the validation folder and reports accuracy stats.
    Expects VAL_DIR to be structured as one subfolder per identity.
    """
    from .ocnn_config import VAL_DIR

    model_path = model_path or (CHECKPOINT_DIR / "best_model.pt")

    recognizer = FaceRecognizer(model_path=model_path)

    total = 0
    correct = 0
    unknown = 0
    wrong = 0

    identity_dirs = sorted([d for d in VAL_DIR.iterdir() if d.is_dir()])

    for identity_dir in tqdm(identity_dirs, desc="Evaluating"):
        true_name = identity_dir.name
        images = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))

        for img_path in images:
            total += 1
            pil_img = Image.open(img_path).convert("RGB")
            pred_name, confidence = recognizer.recognize(pil_img, threshold=threshold)

            if pred_name == "unknown":
                unknown += 1
            elif pred_name == true_name:
                correct += 1
            else:
                wrong += 1

    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"Evaluation complete over {total} images")
    logger.info(f"  Correct  : {correct} ({accuracy:.1%})")
    logger.info(f"  Unknown  : {unknown} ({unknown / total:.1%})")
    logger.info(f"  Wrong    : {wrong}   ({wrong / total:.1%})")

    print(f"\nResults ({total} images, threshold={threshold}):")
    print(f"  Accuracy : {accuracy:.1%}")
    print(f"  Unknown  : {unknown / total:.1%}")
    print(f"  Wrong    : {wrong / total:.1%}")

    return {
        "total": total,
        "correct": correct,
        "unknown": unknown,
        "wrong": wrong,
        "accuracy": accuracy,
    }
