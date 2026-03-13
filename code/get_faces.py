"""
This script is used to fill the data folders with faces from the lfw dataset.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_people


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "data" / "face_recognition_training"
VALIDATION_DIR = PROJECT_ROOT / "data" / "face_recognition_validation"


def safe_name(name: str) -> str:
    """Convert a person name into a filesystem-friendly folder name."""
    return name.strip().lower().replace(" ", "_")


def clear_directory(path: Path) -> None:
    """Remove an existing directory and recreate it."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_image(image_array: np.ndarray, output_path: Path) -> None:
    """
    Save a single LFW image array as an 8-bit grayscale PNG.

    LFW images returned by fetch_lfw_people are typically float arrays.
    """
    # Normalize to 0-255 safely
    image_min = image_array.min()
    image_max = image_array.max()

    if image_max > image_min:
        scaled = (image_array - image_min) / (image_max - image_min)
    else:
        scaled = np.zeros_like(image_array)

    uint8_image = (scaled * 255).astype(np.uint8)
    img = Image.fromarray(uint8_image, mode="L")
    img.save(output_path)


def populate_lfw_folders(
    min_faces_per_person: int = 20,
    resize: float = 0.5,
    validation_images_per_person: int = 2,
) -> None:
    """
    Download LFW and split it into training and validation folders.

    Parameters:
        min_faces_per_person:
            Only include identities with at least this many images.
        resize:
            Resize factor passed to fetch_lfw_people.
        validation_images_per_person:
            Number of images per identity to place in validation.
            Remaining images go to training.
    """
    print("Downloading/loading LFW dataset...")
    lfw_people = fetch_lfw_people(
        min_faces_per_person=min_faces_per_person,
        resize=resize,
    )

    print(f"Loaded {len(lfw_people.images)} images")
    print(f"Loaded {len(lfw_people.target_names)} identities")

    clear_directory(TRAINING_DIR)
    clear_directory(VALIDATION_DIR)

    per_person_counts: dict[str, int] = {}
    train_count = 0
    val_count = 0

    for person_index, person_name in enumerate(lfw_people.target_names):
        folder_name = safe_name(person_name)

        person_train_dir = TRAINING_DIR / folder_name
        person_val_dir = VALIDATION_DIR / folder_name

        person_train_dir.mkdir(parents=True, exist_ok=True)
        person_val_dir.mkdir(parents=True, exist_ok=True)

        # Get indices for this person
        indices = np.where(lfw_people.target == person_index)[0]

        per_person_counts[folder_name] = len(indices)

        for i, image_idx in enumerate(indices):
            image = lfw_people.images[image_idx]
            filename = f"{folder_name}_{i + 1:03d}.png"

            if i < validation_images_per_person:
                output_path = person_val_dir / filename
                val_count += 1
            else:
                output_path = person_train_dir / filename
                train_count += 1

            save_image(image, output_path)

    print("\nDone.")
    print(f"Training images saved:   {train_count}")
    print(f"Validation images saved: {val_count}")
    print(f"Training directory:      {TRAINING_DIR}")
    print(f"Validation directory:    {VALIDATION_DIR}")

    print("\nPer-person image counts:")
    for name, count in sorted(per_person_counts.items()):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    populate_lfw_folders(
        min_faces_per_person=20,
        resize=0.5,
        validation_images_per_person=2,
    )