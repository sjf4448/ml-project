from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import face_recognition
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .classifiers import build_classifier, classifier_display_name
from .config import CLASSIFIER_PATH, ENCODINGS_PATH, TRAINING_DIR, ensure_directories


@dataclass
class TrainingSummary:
    """Simple report object returned after encoding known faces."""

    encoded_faces: int
    processed_images: int
    skipped_images: int
    encodings_path: str


@dataclass
class ClassifierTrainingSummary:
    """Report object returned after fitting a classifier on saved embeddings."""

    classifier_name: str
    training_accuracy: float
    sample_count: int
    class_count: int
    classifier_path: str


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

    def train_classifier(
        self,
        classifier_name: str,
        classifier_path: Path = CLASSIFIER_PATH,
    ) -> ClassifierTrainingSummary:
        """Fit a selectable sklearn classifier from precomputed face embeddings."""
        if not self.encodings_path.exists():
            raise FileNotFoundError(
                f"Encodings file not found at {self.encodings_path}. Run --train first."
            )

        with self.encodings_path.open("rb") as handle:
            payload = pickle.load(handle)

        known_names = np.asarray(payload.get("names", []))
        known_encodings = np.asarray(payload.get("encodings", []), dtype=np.float64)
        if known_names.size == 0 or known_encodings.size == 0:
            raise RuntimeError(
                "Encodings file is empty. Run --train again with usable training images."
            )

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(known_names)

        classifier = build_classifier(classifier_name)
        classifier.fit(known_encodings, y_encoded)
        train_predictions = classifier.predict(known_encodings)
        training_accuracy = float(accuracy_score(y_encoded, train_predictions))

        classifier_path.parent.mkdir(parents=True, exist_ok=True)
        with classifier_path.open("wb") as handle:
            pickle.dump(
                {
                    "classifier_name": classifier_name,
                    "classifier_display_name": classifier_display_name(classifier_name),
                    "classifier": classifier,
                    "label_encoder": label_encoder,
                },
                handle,
            )

        summary = ClassifierTrainingSummary(
            classifier_name=classifier_name,
            training_accuracy=training_accuracy,
            sample_count=int(known_encodings.shape[0]),
            class_count=int(len(label_encoder.classes_)),
            classifier_path=str(classifier_path),
        )
        print(
            "Saved classifier "
            f"'{summary.classifier_name}' with train_accuracy={summary.training_accuracy:.4f} "
            f"({summary.sample_count} samples, {summary.class_count} classes) to {summary.classifier_path}"
        )
        return summary


# Stops infinite loops in edge cases
def main():
    return


if __name__ == "__main__":
    main()
