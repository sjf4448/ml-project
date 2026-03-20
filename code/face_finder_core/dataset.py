"""Dataset utilities for building train/validation folders from LFW."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_people

from .config import KNOWN_DIR, TRAINING_DIR, VALIDATION_DIR


def safe_name(name: str) -> str:
    """Convert a person's display name into a filesystem-safe folder name."""
    return name.strip().lower().replace(" ", "_")


def clear_directory(path: Path) -> None:
    """Delete and recreate a directory so old files do not mix with new splits."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


class LfwDatasetBuilder:
    """Downloads LFW and writes it into this project's folder structure."""

    def __init__(
        self, training_dir: Path = TRAINING_DIR, validation_dir: Path = VALIDATION_DIR
    ):
        self.training_dir = training_dir
        self.validation_dir = validation_dir

    @staticmethod
    def save_image(image_array: np.ndarray, output_path: Path) -> None:
        """Save one grayscale LFW image array as an 8-bit PNG file."""
        image_min = image_array.min()
        image_max = image_array.max()

        if image_max > image_min:
            scaled = (image_array - image_min) / (image_max - image_min)
        else:
            scaled = np.zeros_like(image_array)

        uint8_image = (scaled * 255).astype(np.uint8)
        image = Image.fromarray(uint8_image, mode="L")
        image.save(output_path)

    def populate(
        self,
        min_faces_per_person: int = 20,
        resize: float = 0.5,
        validation_images_per_person: int = 2,
    ) -> None:
        """Build train/validation folders from LFW with a deterministic split."""
        print("Downloading/loading LFW dataset...")
        lfw_people = fetch_lfw_people(
            min_faces_per_person=min_faces_per_person,
            resize=resize,
        )

        print(f"Loaded {len(lfw_people.images)} images")
        print(f"Loaded {len(lfw_people.target_names)} identities")

        clear_directory(self.training_dir)
        clear_directory(self.validation_dir)

        per_person_counts: dict[str, int] = {}
        train_count = 0
        val_count = 0

        for person_index, person_name in enumerate(lfw_people.target_names):
            folder_name = safe_name(person_name)

            person_train_dir = self.training_dir / folder_name
            person_val_dir = self.validation_dir / folder_name
            person_train_dir.mkdir(parents=True, exist_ok=True)
            person_val_dir.mkdir(parents=True, exist_ok=True)

            # Target labels are stored as integer indices; this maps back to each person.
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

                self.save_image(image, output_path)

        print("\nDone.")
        print(f"Training images saved:   {train_count}")
        print(f"Validation images saved: {val_count}")
        print(f"Training directory:      {self.training_dir}")
        print(f"Validation directory:    {self.validation_dir}")

        print("\nPer-person image counts:")
        for name, count in sorted(per_person_counts.items()):
            print(f"  {name}: {count}")


class KnownFacesImporter:
    """Imports known faces and their folders into testing, validation, and training dataset folders"""

    def __init__(
        self,
        training_dir: Path = TRAINING_DIR,
        validation_dir: Path = VALIDATION_DIR,
        known_dir: Path = KNOWN_DIR,
    ):
        self.training_dir = training_dir
        self.validation_dir = validation_dir
        self.known_dir = known_dir

    def import_faces(self, min_faces: int = 3, validation_image_per: int = 1):
        """Imports known faces from the known faces directory"""
        face_path = Path(self.known_dir)
        faces_folder = [folder for folder in face_path.iterdir() if folder.is_dir()]
        VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        train_count = 0
        val_count = 0
        skipped = []
        per_person_counts = {}

        for folder in faces_folder:
            folder_obj = Path(folder)
            included_images = [
                file
                for file in folder_obj.iterdir()
                if file.is_file() and file.suffix in VALID_EXTENSIONS
            ]
            if len(included_images) < min_faces:
                print(
                    f"[WARN]: {folder_obj.name} contains less than {min_faces} faces, skipping"
                )
                skipped.append(folder_obj.name)
                continue

            face_name = safe_name(folder_obj.name)
            training_name = self.training_dir / face_name
            validation_name = self.validation_dir / face_name
            training_name.mkdir(parents=True, exist_ok=True)
            validation_name.mkdir(parents=True, exist_ok=True)

            person_train_count = 0
            person_val_count = 0

            for index, image_path in enumerate(included_images, start=1):
                image = Path(image_path)
                new_image_name = f"{face_name}_{str(index).zfill(3)}{image.suffix}"

                if index <= validation_image_per:
                    new_image_path = validation_name / new_image_name
                    shutil.copy(image, new_image_path)
                    person_val_count += 1
                    val_count += 1
                else:
                    new_image_path = training_name / new_image_name
                    shutil.copy(image, new_image_path)
                    person_train_count += 1
                    train_count += 1

            per_person_counts[face_name] = {
                "train": person_train_count,
                "val": person_val_count,
            }

        print("\nImport Done.")
        print(f"Training images saved:   {train_count}")
        print(f"Validation images saved: {val_count}")
        print(f"Training directory:      {self.training_dir}")
        print(f"Validation directory:    {self.validation_dir}")
        if skipped:
            print(f"Skipped ({len(skipped)}):            {', '.join(skipped)}")
        print("\nPer-person image counts:")
        for name, counts in sorted(per_person_counts.items()):
            print(f"  {name}: {counts['train']} training, {counts['val']} validation")
