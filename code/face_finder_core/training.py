from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import face_recognition
from tqdm import tqdm

from .config import ENCODINGS_PATH, TRAINING_DIR, ensure_directories


@dataclass
class TrainingSummary:
    """Simple report object returned after encoding known faces."""

    encoded_faces: int
    processed_images: int
    skipped_images: int
    encodings_path: str


def _encode_file(filepath: Path, model: str):
    """Required for pickling when using ProcessPoolExecutor"""
    person_label = filepath.parent.name
    image = face_recognition.load_image_file(filepath)
    face_locations = face_recognition.face_locations(image, model=model)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return person_label, face_encodings, len(face_encodings) == 0


class FaceEncoder:
    """Builds a gallery of known face encodings from labeled folders.

    Folder convention:
        data/face_recognition_training/<person_name>/<image files>

    Each subdirectory name is used as that person's identity label.
    """

    def __init__(
        self, training_dir: Path = TRAINING_DIR, encodings_path: Path = ENCODINGS_PATH
    ):
        self.training_dir = training_dir
        self.encodings_path = encodings_path

    def encode_known_faces(self, model: str = "hog") -> TrainingSummary:
        """Extract and save one or more facial embeddings for each training image."""
        ensure_directories()

        names: list[str] = []
        encodings: list[Any] = []
        processed_files = 0
        skipped_files = 0

        all_files = [f for f in self.training_dir.glob("*/*") if f.is_file()]

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(_encode_file, f, model): f for f in all_files}
            for future in tqdm(
                as_completed(futures),
                total=len(all_files),
                desc="Encoding faces",
                unit="img",
            ):
                filepath = futures[future]
                processed_files += 1
                person_label, face_encodings, skipped = future.result()
                if skipped:
                    skipped_files += 1
                    tqdm.write(f"[WARN] No faces found in training image: {filepath}")
                    continue
                for encoding in face_encodings:
                    names.append(person_label)
                    encodings.append(encoding)

        if not encodings:
            raise RuntimeError(
                "No face encodings were created. Check that your training folders contain valid images with visible faces."
            )

        payload = {"names": names, "encodings": encodings}
        with self.encodings_path.open("wb") as handle:
            pickle.dump(payload, handle)

        summary = TrainingSummary(
            encoded_faces=len(encodings),
            processed_images=processed_files,
            skipped_images=skipped_files,
            encodings_path=str(self.encodings_path),
        )
        print(
            f"Saved {summary.encoded_faces} encodings from {summary.processed_images - summary.skipped_images} usable training images to {summary.encodings_path}"
        )
        return summary


# Stops infinite loops in edge cases
def main():
    return


if __name__ == "__main__":
    main()
