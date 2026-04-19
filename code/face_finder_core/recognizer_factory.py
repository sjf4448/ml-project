from pathlib import Path
from .recognition import FaceRecognizer
from .config import ENCODINGS_PATH


class RecognizerFactory:
    def __init__(
        self,
        classifier_path: Path | None,
        encodings_location: Path = ENCODINGS_PATH,
    ):
        self.classifier_path = classifier_path
        self.encodings_location = encodings_location

    def __call__(self):
        return FaceRecognizer(
            encodings_location=self.encodings_location,
            classifier_path=self.classifier_path,
        )