"""
Fine-tunes a pretrained ResNet18/ResNet50 model using the VGGFace2 Dataset and the ArcFace learning method
"""

import logging
import pickle
from pathlib import Path

import torch
import torch.optim as optim
from pytorch_metric_learning.losses import ArcFaceLoss
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import ocnn_mtcnn_align
from .ocnn_config import (
    ARCFACE_MARGIN,
    ARCFACE_SCALE,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EMBEDDING_DIM,
    LOG_INTERVAL,
    LR,
    LR_GAMMA,
    LR_STEP_SIZE,
    MOMENTUM,
    NUM_EPOCHS,
    NUM_WORKERS,
    OUTPUT_DIR,
    SAVE_EVERY,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from .ocnn_dataset import build_dataloaders
from .ocnn_model import build_model

logger = logging.getLogger(__name__)

# Validate directory structure
MIN_IDENTITIES = 10
train_path = Path(TRAIN_DIR)
val_path = Path(VAL_DIR)
test_path = Path(TEST_DIR)
sum_subdirs = (
    sum(1 for x in train_path.iterdir() if x.is_dir())
    + sum(1 for x in val_path.iterdir() if x.is_dir())
    + sum(1 for x in test_path.iterdir() if x.is_dir())
)

if sum_subdirs < MIN_IDENTITIES:
    raise RuntimeError("own_cnn: not enough identities detected!")


# ── LR schedule ───────────────────────────────────────────────────────────────
def build_scheduler(optimizer, num_epochs: int):
    """
    Linear warmup for WARMUP_EPOCHS, then StepLR decay.
    SequentialLR chains them together cleanly.
    """
    warmup = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / WARMUP_EPOCHS)
    step_decay = StepLR(
        optimizer,
        step_size=LR_STEP_SIZE,
        gamma=LR_GAMMA,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, step_decay],
        milestones=[WARMUP_EPOCHS],
    )
    return scheduler


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    device: str,
    epoch: int,
    writer: SummaryWriter,
) -> float:
    model.train()
    loss_fn.train()

    running_loss = 0.0
    total_samples = 0
    global_step = epoch * len(loader)

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1} [train]")):
        if batch is None:
            continue

        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        embeddings = model(images)  # (B, EMBEDDING_DIM)
        loss = loss_fn(embeddings, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if batch_idx % LOG_INTERVAL == 0:
            step = global_step + batch_idx
            writer.add_scalar("Loss/train_step", loss.item(), step)

    epoch_loss = running_loss / max(total_samples, 1)
    return epoch_loss


# ── Validation loop ───────────────────────────────────────────────────────────
def validate_one_epoch(
    model,
    loader,
    loss_fn,
    device: str,
    epoch: int,
) -> float:
    model.eval()
    loss_fn.eval()

    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1} [val]"):
            if batch is None:
                continue

            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            embeddings = model(images)
            loss = loss_fn(embeddings, labels)

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    epoch_loss = running_loss / max(total_samples, 1)
    return epoch_loss


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, scheduler, loss_fn, epoch: int, val_loss: float):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"checkpoint_epoch{epoch + 1:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "loss_fn_state": loss_fn.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )
    logger.info(f"Checkpoint saved → {path}")
    return path


def load_checkpoint(
    path: Path, model, optimizer, scheduler, loss_fn, device: str
) -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    loss_fn.load_state_dict(ckpt["loss_fn_state"])
    logger.info(f"Resumed from {path} (epoch {ckpt['epoch'] + 1})")
    return ckpt["epoch"] + 1  # next epoch to run


# ── Save Embeddings - Inprogress psuedocode ──────────────────────────────────────────────────────────
# TODO: add db saver to ocnn_model.py
def build_embedding_database(model, loader):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, lbls in loader:
            emb = model(images.to(device))
            emb = F.normalize(emb, dim=1)
            embeddings.append(emb.cpu())
            labels.append(lbls)
    embeds, label = torch.cat(embeddings), torch.cat(labels)
    torch.save({"embeddings": embeds, "labels": label}, "face_db.pt")


# ── Main entry ────────────────────────────────────────────────────────────────
def train(
    resume_from: Path | None = None,
    num_epochs: int | None = None,
    batch_size: int | None = None,
):
    _num_epochs = num_epochs if num_epochs is not None else NUM_EPOCHS
    _batch_size = batch_size if batch_size is not None else BATCH_SIZE

    # Trigger mtcnn
    ocnn_mtcnn_align.run()

    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, num_classes, label_encoder = build_dataloaders(
        batch_size=_batch_size,
        num_workers=NUM_WORKERS,
        device=device,
    )
    logger.info(f"Classes: {num_classes}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(device)

    # ── ArcFace loss ──────────────────────────────────────────────────────
    # ArcFaceLoss maintains its own weight matrix (num_classes x embedding_dim)
    # so it needs to know both. It also needs to be on the same device.
    loss_fn = ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=EMBEDDING_DIM,
        margin=ARCFACE_MARGIN,
        scale=int(ARCFACE_SCALE),
    ).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────
    # ArcFace has its own learnable parameters (the class weight matrix),
    # so we include them in the optimizer alongside the model params
    optimizer = optim.SGD(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = build_scheduler(optimizer, NUM_EPOCHS)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    if resume_from is not None:
        start_epoch = load_checkpoint(
            resume_from, model, optimizer, scheduler, loss_fn, device
        )

    # ── Tensorboard ───────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(CHECKPOINT_DIR / "runs"))

    # ── Save label encoder alongside checkpoints ──────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    label_encoder_path = OUTPUT_DIR / "label_encoder.pkl"
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    # ── Loop ──────────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(start_epoch, _num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch, writer
        )
        val_loss = validate_one_epoch(model, val_loader, loss_fn, device, epoch)
        scheduler.step()

        logger.info(
            f"Epoch {epoch + 1}/{_num_epochs} — "
            f"train loss: {train_loss:.4f}  val loss: {val_loss:.4f}  "
            f"lr: {scheduler.get_last_lr()[0]:.6f}"
        )

        writer.add_scalars(
            "Loss/epoch",
            {
                "train": train_loss,
                "val": val_loss,
            },
            epoch,
        )
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # ── Checkpoint ────────────────────────────────────────────────────
        if (epoch + 1) % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, scheduler, loss_fn, epoch, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = CHECKPOINT_DIR / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            logger.info(f"New best model saved (val loss: {val_loss:.4f})")

    writer.close()
    logger.info("Training complete.")
    return model, label_encoder
