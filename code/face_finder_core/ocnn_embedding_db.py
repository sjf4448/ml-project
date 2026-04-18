"""
embedding_db.py — builds and saves the face embedding database after training.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

from .ocnn_config import IMAGE_SIZE, KNOWN_DIR, MTCNN_MARGIN, MTCNN_MIN_FACE, OUTPUT_DIR
from .ocnn_dataset import get_val_transforms

logger = logging.getLogger(__name__)


# ── Aggregation ───────────────────────────────────────────────────────────────


def aggregate_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each identity, average all its embeddings into one representative
    vector and re-normalize. Reduces the database size and smooths out
    per-image noise.
    """
    unique_labels = labels.unique()
    agg_embeddings = []
    agg_labels = []

    for lbl in unique_labels:
        mask = labels == lbl
        mean_emb = embeddings[mask].mean(dim=0)
        mean_emb = F.normalize(mean_emb, dim=0)
        agg_embeddings.append(mean_emb)
        agg_labels.append(lbl)

    return torch.stack(agg_embeddings), torch.stack(agg_labels)


# ── Encode Known Faces ────────────────────────────────────────────────────────
def encode_known_faces(
    model, device: str, label_encoder, known_dir: Path | None = None
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Encodes all images in known_faces/<name> - returns theirs embeddings and integer labels
    """
    known_dir = known_dir or KNOWN_DIR
    if not known_dir.exists():
        logger.info("No known_faces dir found, skipping")
        return None

    identity_dirs = [d for d in known_dir.iterdir() if d.is_dir()]
    if not identity_dirs:
        logger.info("known_faces/ is empty, skipping")
        return None

    mtcnn = MTCNN(
        image_size=IMAGE_SIZE,
        margin=MTCNN_MARGIN,
        min_face_size=MTCNN_MIN_FACE,
        keep_all=False,
        post_process=False,
        device=device,
    )
    transform = get_val_transforms()
    model.eval()

    # ── Extend label encoder with any new identities ───────────────────────
    existing = list(label_encoder.classes_)
    new_names = [d.name for d in identity_dirs if d.name not in existing]
    if new_names:
        import numpy as np

        label_encoder.classes_ = np.array(sorted(existing + new_names))
        logger.info(
            f"Added {len(new_names)} new identities to label encoder: {new_names}"
        )

    all_embeddings = []
    all_labels = []

    for identity_dir in tqdm(identity_dirs, desc="Encoding known faces"):
        name = identity_dir.name
        label = int(label_encoder.transform([name])[0])
        images = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))

        for img_path in images:
            pil_img = Image.open(img_path).convert("RGB")

            # Resize if provided image is too small/large
            w, h = pil_img.size
            min_dim = min(w, h)
            max_dim = max(w, h)

            if min_dim < IMAGE_SIZE:
                # too small — upscale so MTCNN has enough pixels to detect
                scale = IMAGE_SIZE / min_dim
                pil_img = pil_img.resize(
                    (int(w * scale), int(h * scale)), Image.BILINEAR
                )
            elif max_dim > 1920:
                # excessively large — downscale to save memory
                scale = 1920 / max_dim
                pil_img = pil_img.resize(
                    (int(w * scale), int(h * scale)), Image.BILINEAR
                )

            face_tensor = mtcnn(pil_img)
            if face_tensor is None:
                logger.debug(f"No face detected in {img_path.name}, skipping")
                continue

            face_np = face_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
            augmented = transform(image=face_np)
            tensor = augmented["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model(tensor).cpu()

            all_embeddings.append(emb)
            all_labels.append(label)

    if not all_embeddings:
        logger.warning("No faces detected in known_faces/, skipping")
        return None

    return torch.cat(all_embeddings, dim=0), torch.tensor(all_labels, dtype=torch.long)


# ── Save embeddings as pkl files ──────────────────────────────────────────────
def save_encodings_pkl(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    label_encoder,
    path: Path | None = None,
):
    """
    Saves embeddings in the encodings.pkl format expected by the existing
    face_finder.py pipeline — compatible with classifiers.py and distance matching.
    """
    path = path or (OUTPUT_DIR / "encodings.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy — sklearn classifiers expect numpy arrays, not tensors
    emb_np = embeddings.cpu().numpy()  # (N, 512)
    labels_np = label_encoder.inverse_transform(labels.cpu().numpy())  # string names

    payload = {
        "encodings": emb_np,
        "names": labels_np.tolist(),
    }

    with open(path, "wb") as f:
        pickle.dump(payload, f)

    logger.info(f"encodings.pkl saved → {path} ({len(labels_np)} entries)")


# ── Build ─────────────────────────────────────────────────────────────────────


def build_embedding_database(
    model,
    loader,
    device: str,
    aggregate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Runs the full dataloader through the model, collects all embeddings,
    and optionally aggregates per identity into a single mean embedding.

    Returns:
        embeddings : (N, EMBEDDING_DIM) float32 tensor
        labels     : (N,) int64 tensor
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Building embedding DB"):
            if batch is None:
                continue

            images, labels = batch
            images = images.to(device, non_blocking=True)

            embeddings = model(images)  # already L2-normalized
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)  # (N, D)
    labels = torch.cat(all_labels, dim=0)  # (N,)

    if aggregate:
        embeddings, labels = aggregate_embeddings(embeddings, labels)
        logger.info(f"Aggregated to {len(labels)} identity embeddings")

    return embeddings, labels


# ── Save / load ───────────────────────────────────────────────────────────────


def save_embedding_database(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    label_encoder,
    path: Path | None = None,
):
    path = path or (OUTPUT_DIR / "face_db.pt")
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({"embeddings": embeddings, "labels": labels}, path)
    logger.info(f"Embedding DB saved → {path}")

    # save label encoder alongside so recognition.py can load both together
    encoder_path = path.parent / "label_encoder.pkl"
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    logger.info(f"Label encoder saved → {encoder_path}")


def load_embedding_database(
    path: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    path = path or (OUTPUT_DIR / "face_db.pt")
    data = torch.load(path)
    return data["embeddings"], data["labels"]


# ── Convenience wrapper ───────────────────────────────────────────────────────


def build_and_save_database(
    model, loader, device: str, label_encoder, aggregate: bool = True
):
    logger.info("Building embedding database...")
    embeddings, labels = build_embedding_database(
        model, loader, device, aggregate=False
    )  # no aggregation for pkl — one row per image

    # ── Merge known faces ──────────────────────────────────────────────────
    known_result = encode_known_faces(model, device, label_encoder)
    if known_result is not None:
        known_embeddings, known_labels = known_result
        embeddings = torch.cat([embeddings, known_embeddings], dim=0)
        labels = torch.cat([labels, known_labels], dim=0)
        logger.info(f"Merged {len(known_labels)} known face embeddings into database")

    save_encodings_pkl(embeddings, labels, label_encoder)  # existing pipeline

    if aggregate:
        embeddings, labels = aggregate_embeddings(embeddings, labels)

    save_embedding_database(embeddings, labels, label_encoder)  # faiss/torch pipeline

    return embeddings, labels
