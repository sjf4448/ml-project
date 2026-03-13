"""Core modules for the educational face-finder pipeline.

The package is split by responsibility so students can inspect one stage at a time:
- dataset.py   -> prepare training/validation folders from LFW
- training.py  -> encode known people
- recognition.py -> detect and recognize faces in a single image
- validation.py  -> batch evaluation on a validation folder
- webcam.py    -> capture image from webcam and run recognition
- cli.py       -> command-line entry point
"""

from .config import (
    ANNOTATED_DIR,
    CROPS_DIR,
    ENCODINGS_PATH,
    METADATA_DIR,
    OUTPUT_DIR,
    TRAINING_DIR,
    VALIDATION_DIR,
    ensure_directories,
)
from .recognition import DetectionResult, FaceRecognizer
from .training import FaceEncoder
from .validation import ValidationRunner

__all__ = [
    "ANNOTATED_DIR",
    "CROPS_DIR",
    "DetectionResult",
    "ENCODINGS_PATH",
    "FaceEncoder",
    "FaceRecognizer",
    "METADATA_DIR",
    "OUTPUT_DIR",
    "TRAINING_DIR",
    "VALIDATION_DIR",
    "ValidationRunner",
    "ensure_directories",
]

