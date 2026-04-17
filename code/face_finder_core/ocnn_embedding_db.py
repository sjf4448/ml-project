"""
embedding_db.py — builds and saves the face embedding database after training.
"""

import logging
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .ocnn_config import OUTPUT_DIR

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

    save_encodings_pkl(embeddings, labels, label_encoder)  # existing pipeline

    if aggregate:
        embeddings, labels = aggregate_embeddings(embeddings, labels)

    save_embedding_database(embeddings, labels, label_encoder)  # faiss/torch pipeline

    return embeddings, labels
