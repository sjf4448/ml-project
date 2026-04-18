# face_finder_core/model.py

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ocnn_config import BACKBONE, EMBEDDING_DIM, PRETRAINED


class FaceEmbedder(nn.Module):
    """
    ResNet-18 backbone from timm with the classification head replaced
    by an embedding layer. Output embeddings are L2-normalized so they
    live on the unit hypersphere, which is what ArcFace expects.
    """

    def __init__(self):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────
        # num_classes=0 tells timm to remove the classifier head entirely
        # and return the pooled feature vector instead
        self.backbone = timm.create_model(
            BACKBONE,
            pretrained=PRETRAINED,
            num_classes=0,  # strips the FC head
        )

        backbone_out_dim = self.backbone.num_features  # 512 for ResNet-18

        # ── Embedding head ────────────────────────────────────────────────
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_out_dim, EMBEDDING_DIM),  # type: ignore
            nn.BatchNorm1d(EMBEDDING_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # (B, 512)
        embeddings = self.embedding_head(features)  # (B, EMBEDDING_DIM)
        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalize
        return embeddings


def build_model(device: str) -> FaceEmbedder:
    model = FaceEmbedder()
    model = model.to(device)
    return model
