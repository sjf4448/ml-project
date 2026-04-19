from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from .config import VALIDATION_DIR, UNKNOWN_DIR, ensure_directories
from .recognition import FaceRecognizer


class ValidationRunner:
    """Batch runner that applies recognition to every validation image."""

    def __init__(
        self, recognizer: FaceRecognizer, validation_dir: Path = VALIDATION_DIR, unknown_faces_dir: Path = UNKNOWN_DIR
    ):
        self.recognizer = recognizer
        self.validation_dir = validation_dir
        self.unknown_faces_dir = unknown_faces_dir

    def run(self, model: str = "hog", tolerance: float = 0.6) -> None:
        ensure_directories()
        VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        validation_files = [
            path
            for path in self.validation_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        ]

        validation_files.extend([
            path
            for path in self.unknown_faces_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        ])

        if not validation_files:
            print(f"No validation images found in {self.validation_dir}")
            return

        for image_path in tqdm(validation_files, desc="Validating", unit="img"):
            tqdm.write(f"[VALIDATE] {image_path}")
            self.recognizer.recognize_faces(
                image_location=str(image_path),
                model=model,
                tolerance=tolerance,
                save_output=True,
                show_image=False,
            )

    def run_deterministic(self, model="hog", tolerance=0.6):
        VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

        files = [
            p for p in self.validation_dir.rglob("*")
            if p.suffix.lower() in VALID_EXTENSIONS
        ]

        files += [
            p for p in self.unknown_faces_dir.rglob("*")
            if p.suffix.lower() in VALID_EXTENSIONS
        ]

        results = []

        for path in files:
            actual = path.parent.name
            if actual == "unknown_faces":
                actual = "Unknown"

            detections = self.recognizer.recognize_faces(
                image_location=str(path),
                model=model,
                tolerance=tolerance,
                save_output=False,
                show_image=False,
            )


            for det in detections:
                results.append({
                    "actual_name": actual,
                    "detected_name": det.detected_name,
                    "confidence_distance": det.confidence_distance,
                    "all_distances": det.all_distances,
                })

        return results
