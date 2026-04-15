"""embeddingg db file - in progress"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"


# Compress mutiple images into a embedding before saving
def aggregate_embeddings(embeddings, labels):
    unique_labels = labels.unique()
    agg_embeddings = []
    agg_labels = []

    for lbl in unique_labels:
        mask = labels == lbl
        mean_emb = embeddings[mask].mean(dim=0)
        mean_emb = F.normalize(mean_emb, dim=0)

        agg_embeddings.append(mean_emb)
        agg_labels.append(lbl)

    return torch.stack(agg_embeddings), torch.tensor(agg_labels)


def build_embedding_database(model, loader, device):
    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Building embedding DB"):
            if batch is None:
                continue

            images, labels = batch
            images = images.to(device)

            embeddings = model(images)  # already normalized
            embeddings = embeddings.detach().cpu()

            all_embeddings.append(embeddings)
            all_labels.append(labels)
    aggregate_embeddings(all_embeddings, all_labels)
    return torch.cat(all_embeddings), torch.cat(all_labels)


# After training loop
logger.info("Building embedding database...")

db_embeddings, db_labels = build_embedding_database(
    model,
    train_loader,  # or combine train + val
    device,
)

db_path = OUTPUT_DIR / "face_db.pt"

torch.save(
    {
        "embeddings": db_embeddings,
        "labels": db_labels,
    },
    db_path,
)

logger.info(f"Embedding DB saved → {db_path}")
