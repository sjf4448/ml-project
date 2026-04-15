import os
import random
import shutil
from pathlib import Path

import kagglehub
from tqdm import tqdm

TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
MIN_IMAGES = 10
MAX_IDENTITIES = 500
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "face_recognition_training"
VAL_DIR = PROJECT_ROOT / "data" / "face_recognition_validation"
TEST_DIR = PROJECT_ROOT / "data" / "face_recognition_test"

# ---------- Downloading dataset
print("Downloading VGGFace2 Dataset...")
vgg_down_path = kagglehub.dataset_download("hearfool/vggface2")
vgg_down_path = Path(vgg_down_path)
print(f"Download Dir:{vgg_down_path}")


# ------------ Collect Identities
def collect_identities(root: Path) -> dict[str, list[Path]]:
    """Returns a dict of {identity_id: [image_path, ...]} from a VGGFace2 folder"""
    identities = {}
    for identity_dir in sorted(root.iterdir()):
        if not identity_dir.is_dir():
            continue
        images = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))
        if len(images) >= MIN_IMAGES:
            identities[identity_dir.name] = images
    return identities


all_identities = collect_identities(vgg_down_path)

for split_name in ("train", "test"):
    split_dir = vgg_root / split_name
    if split_dir.exists():
        print(f"Scanning {split_dir}...")
        found = collect_identities(split_dir)
        for identity, images in found.items():
            all_identities.setdefault(identity, []).extend(images)

print(f"Total identities found (>= {MIN_IMAGES} images): {len(all_identities)}")

# --------------- Subset and clear destinations

random.seed(SEED)
identity_list = sorted(all_identities.keys())

if MAX_IDENTITIES is not None:
    identity_list = random.sample(
        identity_list, min(MAX_IDENTITIES, len(identity_list))
    )
    print(f"Subsetting to {len(identity_list)} identities")

print("Clearing destination folders...")

for dest_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
        print(f"  Cleared {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created {dest_dir}")


# ------------------- Copy images to dirs
def copy_images(images: list[Path], dest_dir: Path, identity: str):
    out = dest_dir / identity
    out.mkdir(parents=True, exist_ok=True)
    for img in images:
        shutil.copy2(img, out / img.name)


print("Splitting and copying images...")

train_count = val_count = test_count = 0

for identity in tqdm(identity_list):
    images = all_identities[identity].copy()
    random.shuffle(images)

    n = len(images)
    n_train = max(1, int(n * TRAIN_SPLIT))
    n_val = max(1, int(n * VAL_SPLIT))
    # remainder goes to test

    train_imgs = images[:n_train]
    val_imgs = images[n_train : n_train + n_val]
    test_imgs = images[n_train + n_val :]

    copy_images(train_imgs, TRAIN_DIR, identity)
    copy_images(val_imgs, VAL_DIR, identity)
    if test_imgs:
        copy_images(test_imgs, TEST_DIR, identity)

    train_count += len(train_imgs)
    val_count += len(val_imgs)
    test_count += len(test_imgs)

print("\nDone.")
print(f"  Train : {train_count:,} images across {len(identity_list)} identities")
print(f"  Val   : {val_count:,} images")
print(f"  Test  : {test_count:,} images")
print(f"\n  → {TRAIN_DIR}")
print(f"  → {VAL_DIR}")
print(f"  → {TEST_DIR}")
