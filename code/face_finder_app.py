from __future__ import annotations

from face_finder_core.recognition import FaceRecognizer
from face_finder_core.webcam import WebcamCaptureSession


def main() -> None:
    """Launch the webcam app using the class-based core implementation."""
    recognizer = FaceRecognizer()
    session = WebcamCaptureSession(recognizer=recognizer)
    session.run()


if __name__ == "__main__":
    main()
