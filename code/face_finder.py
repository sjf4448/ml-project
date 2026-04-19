"""Backward-compatible entry script for the face-finder CLI.

The implementation now lives in `face_finder_core/` and is split into classes
to make the pipeline easier to study. This file keeps old imports working.
"""

from __future__ import annotations

from pathlib import Path

from face_finder_core import (
    ANNOTATED_DIR,
    CLASSIFIER_PATH,
    CROPS_DIR,
    ENCODINGS_PATH,
    METADATA_DIR,
    OUTPUT_DIR,
    TRAINING_DIR,
    VALIDATION_DIR,
    DetectionResult,
    FaceEncoder,
    FaceRecognizer,
    ValidationRunner,
    ensure_directories,
)
from face_finder_core.cli import build_parser, main
from face_finder_core.hyperparameters import get_tolerance

__all__ = [
    "ANNOTATED_DIR",
    "CLASSIFIER_PATH",
    "CROPS_DIR",
    "DetectionResult",
    "ENCODINGS_PATH",
    "FaceEncoder",
    "FaceRecognizer",
    "METADATA_DIR",
    "OUTPUT_DIR",
    "TRAINING_DIR",
    "VALIDATION_DIR",
    "build_parser",
    "encode_known_faces",
    "ensure_directories",
    "main",
    "recognize_faces",
    "validate",
]


def encode_known_faces(model: str = "hog", encodings_location: Path = ENCODINGS_PATH) -> None:
    """Compatibility function that delegates to `FaceEncoder`."""
    FaceEncoder(encodings_path=encodings_location).encode_known_faces(model=model)


def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = ENCODINGS_PATH,
    tolerance: float = get_tolerance(),
    save_output: bool = True,
    show_image: bool = False,
) -> list[DetectionResult]:
    """Compatibility function that delegates to `FaceRecognizer`."""
    recognizer = FaceRecognizer(encodings_location=encodings_location)
    return recognizer.recognize_faces(
        image_location=image_location,
        model=model,
        tolerance=tolerance,
        save_output=save_output,
        show_image=show_image,
    )


def validate(model: str = "hog", tolerance: float = get_tolerance()) -> None:
    """Compatibility function that delegates to `ValidationRunner`."""
    recognizer = FaceRecognizer()
    ValidationRunner(recognizer=recognizer).run(model=model, tolerance=tolerance)


if __name__ == "__main__":
    main()
