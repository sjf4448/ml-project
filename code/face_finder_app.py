from __future__ import annotations

import argparse

from face_finder_core.recognition import FaceRecognizer
from face_finder_core.webcam import WebcamCaptureSession


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
        default=0.6,
        help="Recognition tolerance. Lower is stricter. Typical values: 0.45 to 0.6",
    )
    return parser


def main() -> None:
    """Launch the webcam app using the class-based core implementation."""
    args = build_parser().parse_args()

    if args.camera_index < 0:
        raise ValueError("--camera-index must be 0 or greater")

    if not 0.0 <= args.tolerance <= 1.0:
        raise ValueError("--tolerance must be between 0.0 and 1.0")

    recognizer = FaceRecognizer()
    session = WebcamCaptureSession(
        recognizer=recognizer,
        tolerance=args.tolerance,
        camera_index=args.camera_index,
    )
    session.run()


if __name__ == "__main__":
    main()
