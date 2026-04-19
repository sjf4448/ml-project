from __future__ import annotations

from pathlib import Path

# Compute paths relative to the repository root so scripts run from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "data" / "face_recognition_training"
VALIDATION_DIR = PROJECT_ROOT / "data" / "face_recognition_validation"
TEST_DIR = PROJECT_ROOT / "data" / "face_recognition_test"
OUTPUT_DIR = PROJECT_ROOT / "data" / "face_recognition_output"
KNOWN_DIR = PROJECT_ROOT / "data" / "known_faces"
UNKNOWN_DIR = PROJECT_ROOT / "data" / "unknown_faces"
ENCODINGS_PATH = OUTPUT_DIR / "encodings.pkl"
CLASSIFIER_PATH = OUTPUT_DIR / "classifier.pkl"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
CROPS_DIR = OUTPUT_DIR / "crops"
METADATA_DIR = OUTPUT_DIR / "metadata"

HYPERPARAMETERS_PATH = PROJECT_ROOT / "data" / "hyperparameters.json"

BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"


def ensure_directories() -> None:
    """Create all folders used by the pipeline if they do not exist yet."""
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    HYPERPARAMETERS_PATH.touch()
