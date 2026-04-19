from __future__ import annotations

import argparse
from pathlib import Path

from face_finder_core.config import CLASSIFIER_PATH
from face_finder_core.recognition import FaceRecognizer
from face_finder_core.webcam import WebcamCaptureSession
from face_finder_core.hyperparameters import get_tolerance


def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for the webcam app."""
    parser = argparse.ArgumentParser(description="Webcam face recognition app")
    parser.add_argument(
        "-c",
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index to open (try 0, 1, or 2 on macOS)",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=get_tolerance(),
        help="Recognition tolerance. Lower is stricter. Typical values: 0.45 to 0.6",
    )
    parser.add_argument(
        "--classifier-path",
        type=str,
        default=str(CLASSIFIER_PATH),
        help="Classifier artifact path created by face_finder.py --train-classifier",
    )
    parser.add_argument(
        "--disable-classifier",
        action="store_true",
        help="Use distance-only recognition even if a classifier artifact exists",
    )
    return parser


def main() -> None:
    """Launch the webcam app using the class-based core implementation."""
    args = build_parser().parse_args()

    if args.camera_index < 0:
        raise ValueError("--camera-index must be 0 or greater")

    if not 0.0 <= args.tolerance <= 1.0:
        raise ValueError("--tolerance must be between 0.0 and 1.0")

    classifier_path = None if args.disable_classifier else Path(args.classifier_path)
    recognizer = FaceRecognizer(classifier_path=classifier_path)
    session = WebcamCaptureSession(
        recognizer=recognizer,
        tolerance=args.tolerance,
        camera_index=args.camera_index,
    )
    print("Tolerance set to:", args.tolerance)
    session.run()


if __name__ == "__main__":
    main()
