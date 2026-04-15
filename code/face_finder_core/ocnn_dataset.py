# face_finder_core/dataset.py

import logging
import pickle
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from facenet_pytorch import MTCNN
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from .ocnn_config import (
    AUGMENT_PROB,
    COLOR_JITTER,
    ENCODINGS_PATH,
    IMAGE_SIZE,
    MTCNN_MARGIN,
    MTCNN_MIN_FACE,
    TRAIN_DIR,
    VAL_DIR,
)

logger = logging.getLogger(__name__)

# ── MTCNN singleton ───────────────────────────────────────────────────────────
# One shared instance — initializing it per-sample is very slow


def get_mtcnn(device: str = "cpu") -> MTCNN:
    return MTCNN(
        image_size=IMAGE_SIZE,
        margin=MTCNN_MARGIN,
        min_face_size=MTCNN_MIN_FACE,
        keep_all=False,  # only the most prominent face per image
        post_process=False,  # return raw pixels, not whitened
        device=device,
    )


# ── Transforms ────────────────────────────────────────────────────────────────


def get_train_transforms() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=COLOR_JITTER,
                contrast=COLOR_JITTER,
                saturation=COLOR_JITTER,
                hue=0.0,
                p=AUGMENT_PROB,
            ),
            A.GaussianBlur(blur_limit=(3, 3), p=0.1),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # type: ignore
            ToTensorV2(),
        ]  # type: ignore
    )


def get_val_transforms() -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # type: ignore
            ToTensorV2(),
        ]  # type : ignore
    )


# ── Dataset ───────────────────────────────────────────────────────────────────


class VGGFace2Dataset(Dataset):
    """
    Loads pre-cropped face images from a directory structured as:
        root/
            identity_a/
                img1.jpg
                img2.jpg
            identity_b/
                ...

    MTCNN alignment is applied per-sample. Images that MTCNN fails to
    detect a face in are skipped during __init__ and not returned.
    """

    def __init__(
        self,
        root: Path,
        transform: A.Compose,
        mtcnn: MTCNN,
        label_encoder: LabelEncoder | None = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.mtcnn = mtcnn

        self.samples: list[tuple[Path, int]] = []

        # ── Gather all image paths and identity labels ─────────────────────
        identity_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        raw_labels = [d.name for d in identity_dirs]

        # ── Fit or reuse label encoder ─────────────────────────────────────
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(raw_labels)
        else:
            self.label_encoder = label_encoder

        # ── Build sample list ──────────────────────────────────────────────
        skipped = 0
        for identity_dir in identity_dirs:
            if identity_dir.name not in self.label_encoder.classes_:  # type: ignore
                continue
            label = int(self.label_encoder.transform([identity_dir.name])[0])
            for img_path in sorted(identity_dir.glob("*.jpg")):
                self.samples.append((img_path, label))
            for img_path in sorted(identity_dir.glob("*.png")):
                self.samples.append((img_path, label))

        logger.info(
            f"Dataset loaded from {self.root}: "
            f"{len(self.samples)} images, "
            f"{len(self.label_encoder.classes_)} identities, "  # type: ignore
            f"{skipped} skipped"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int] | None:
        img_path, label = self.samples[idx]

        # ── Load image ─────────────────────────────────────────────────────
        pil_img = Image.open(img_path).convert("RGB")

        # ── MTCNN alignment ────────────────────────────────────────────────
        # Returns a (C, H, W) float tensor if a face is found, else None
        face_tensor = self.mtcnn(pil_img)

        if face_tensor is None:
            # No face detected — return a black tensor as a safe fallback.
            # The collate_fn below filters these out before they hit the model.
            return None

        # face_tensor is float32 in [0, 255] from MTCNN (post_process=False)
        # Convert to uint8 numpy for albumentations
        face_np = face_tensor.permute(1, 2, 0).numpy().astype(np.uint8)

        # ── Augmentation + normalization ───────────────────────────────────
        augmented = self.transform(image=face_np)
        tensor = augmented["image"]  # (C, H, W), normalized

        return tensor, label

    @property
    def num_classes(self) -> int:
        return len(self.label_encoder.classes_)  # type: ignore

    def save_label_encoder(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.label_encoder, f)
        logger.info(f"Label encoder saved to {path}")


# ── Collate ───────────────────────────────────────────────────────────────────


def collate_skip_nones(
    batch: list[tuple[torch.Tensor, int] | None],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Filters out None entries (images where MTCNN found no face),
    then stacks the rest into a batch. Returns None if the entire
    batch is empty so the training loop can skip it cleanly.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    tensors, labels = zip(*batch)
    return torch.stack(tensors), torch.tensor(labels, dtype=torch.long)


# ── Factory ───────────────────────────────────────────────────────────────────


def build_dataloaders(
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple:
    """
    Returns (train_loader, val_loader, num_classes, label_encoder).
    The label encoder fitted on train is reused for val so classes are consistent.
    """
    mtcnn = get_mtcnn(device)

    train_dataset = VGGFace2Dataset(
        root=TRAIN_DIR,
        transform=get_train_transforms(),
        mtcnn=mtcnn,
    )

    val_dataset = VGGFace2Dataset(
        root=VAL_DIR,
        transform=get_val_transforms(),
        mtcnn=mtcnn,
        label_encoder=train_dataset.label_encoder,  # reuse fitted encoder
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_skip_nones,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_skip_nones,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        train_dataset.num_classes,
        train_dataset.label_encoder,
    )
