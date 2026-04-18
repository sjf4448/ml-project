import random
import shutil
from pathlib import Path

import kagglehub
from tqdm import tqdm

from .ocnn_config import TEST_DIR, TRAIN_DIR, VAL_DIR

TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
MIN_IMAGES = 10
MAX_IDENTITIES = 500
SEED = 42


def collect_identities(root: Path, min_images: int) -> dict[str, list[Path]]:
    identities = {}
    for identity_dir in sorted(root.iterdir()):
        if not identity_dir.is_dir():
            continue
        images = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))
        if len(images) >= min_images:
            identities[identity_dir.name] = images
    return identities


def copy_images(images: list[Path], dest_dir: Path, identity: str):
    out = dest_dir / identity
    out.mkdir(parents=True, exist_ok=True)
    for img in images:
        shutil.copy2(img, out / img.name)


def run(
    max_identities: int | None = MAX_IDENTITIES,
    min_images: int = MIN_IMAGES,
    clear: bool = True,
):
    random.seed(SEED)

    print("Downloading VGGFace2 Dataset...")
    vgg_down_path = Path(kagglehub.dataset_download("hearfool/vggface2"))
    print(f"Download Dir: {vgg_down_path}")

    all_identities: dict[str, list[Path]] = collect_identities(
        vgg_down_path, min_images
    )
    for split_name in ("train", "test"):
        split_dir = vgg_down_path / split_name
        if split_dir.exists():
            print(f"Scanning {split_dir}...")
            found = collect_identities(split_dir, min_images)
            for identity, images in found.items():
                all_identities.setdefault(identity, []).extend(images)

    print(f"Total identities found (>= {min_images} images): {len(all_identities)}")

    identity_list = sorted(all_identities.keys())
    if max_identities is not None:
        identity_list = random.sample(
            identity_list, min(max_identities, len(identity_list))
        )
        print(f"Subsetting to {len(identity_list)} identities")

    print("Clearing destination folders...")
    for dest_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if clear and dest_dir.exists():
            shutil.rmtree(dest_dir)
            print(f"  Cleared {dest_dir}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created {dest_dir}")

    print("Splitting and copying images...")
    train_count = val_count = test_count = 0

    for identity in tqdm(identity_list):
        images = all_identities[identity].copy()
        random.shuffle(images)
        n = len(images)
        n_train = max(1, int(n * TRAIN_SPLIT))
        n_val = max(1, int(n * VAL_SPLIT))

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
    print(f"\n  → {TRAIN_DIR}\n  → {VAL_DIR}\n  → {TEST_DIR}")
