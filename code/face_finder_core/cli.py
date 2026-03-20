from __future__ import annotations

import argparse

from .recognition import FaceRecognizer
from .training import FaceEncoder
from .validation import ValidationRunner
from .generate_statistics import generate_statistics

def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for training, validation, and single-image testing."""
    parser = argparse.ArgumentParser(
        description="Face detection and recognition pipeline for known-person classification"
    )
    parser.add_argument("--train", action="store_true", help="Encode labeled training faces")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run recognition on every image in validation",
    )
    parser.add_argument("--test", action="store_true", help="Run recognition on one image")
    parser.add_argument("-f", "--file", dest="file", help="Path to a single image for testing")
    parser.add_argument(
        "-m",
        "--model",
        default="hog",
        choices=["hog", "cnn"],
        help="Face detector backend: hog for CPU, cnn for GPU",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=0.6,
        help="Recognition tolerance. Lower is stricter. Typical values: 0.45 to 0.6",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated image after processing",
    )
    parser.add_argument(
        "--statistics",
        action="store_true",
        help="Generate statistics about the validation"
    )
    return parser


def main() -> None:
    """Run selected stages of the pipeline from command-line flags."""
    parser = build_parser()
    args = parser.parse_args()

    encoder = FaceEncoder()
    recognizer = FaceRecognizer()
    validator = ValidationRunner(recognizer=recognizer)

    if args.train:
        encoder.encode_known_faces(model=args.model)

    if args.validate:
        validator.run(model=args.model, tolerance=args.tolerance)

    if args.test:
        if not args.file:
            parser.error("--test requires --file PATH")
        recognizer.recognize_faces(
            image_location=args.file,
            model=args.model,
            tolerance=args.tolerance,
            save_output=True,
            show_image=args.show,
        )
        
    if args.statistics:
        generate_statistics()

    if not any([args.train, args.validate, args.test, args.statistics]):
        parser.print_help()

