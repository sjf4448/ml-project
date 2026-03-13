from __future__ import annotations

from pathlib import Path

from .config import VALIDATION_DIR, ensure_directories
from .recognition import FaceRecognizer


class ValidationRunner:
    """Batch runner that applies recognition to every validation image."""

    def __init__(self, recognizer: FaceRecognizer, validation_dir: Path = VALIDATION_DIR):
        self.recognizer = recognizer
        self.validation_dir = validation_dir

    def run(self, model: str = "hog", tolerance: float = 0.6) -> None:
        ensure_directories()

        validation_files = [path for path in self.validation_dir.rglob("*") if path.is_file()]
        if not validation_files:
            print(f"No validation images found in {self.validation_dir}")
            return

        for image_path in validation_files:
            print(f"\n[VALIDATE] {image_path}")
            self.recognizer.recognize_faces(
                image_location=str(image_path),
                model=model,
                tolerance=tolerance,
                save_output=True,
                show_image=False,
            )

