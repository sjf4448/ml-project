from __future__ import annotations

import argparse
from pathlib import Path
from .hyperparameter_trainer import train_tolerance_hyperparameter

from .classifiers import available_classifier_names
from .config import CLASSIFIER_PATH
from .recognition import FaceRecognizer
from .training import FaceEncoder
from .validation import ValidationRunner
from .generate_statistics import generate_statistics
from .recognizer_factory import RecognizerFactory
from .hyperparameters import get_tolerance

def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for training, validation, and single-image testing."""
    parser = argparse.ArgumentParser(
        description="Face detection and recognition pipeline for known-person classification"
    )
    parser.add_argument("--train", action="store_true", help="Encode labeled training faces")
    parser.add_argument(
        "--train-classifier",
        action="store_true",
        help="Train a classifier from saved embeddings for runtime recognition",
    )
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
        default=get_tolerance(),
        help="Recognition tolerance. Lower is stricter. Typical values: 0.45 to 0.6",
    )
    parser.add_argument(
        "--classifier",
        default="linear_svc",
        choices=available_classifier_names(),
        help="Classifier to train/use (choose after running model_comparison.py)",
    )
    parser.add_argument(
        "--classifier-path",
        type=str,
        default=str(CLASSIFIER_PATH),
        help="Path to classifier artifact created by --train-classifier",
    )
    parser.add_argument(
        "--disable-classifier",
        action="store_true",
        help="Disable classifier inference and use distance-only recognition",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated image after processing",
    )
    parser.add_argument(
        "--statistics",
        action="store_true",
        help="Generate statistics about the validation (run after --validate to create the necessary data)",
    )
    parser.add_argument(
        "--train-hyperparameter",
        action="store_true",
        help="Train the tolerance hyperparameter using the validation set",
    )
    return parser


def main() -> None:
    """Run selected stages of the pipeline from command-line flags."""
    parser = build_parser()
    args = parser.parse_args()

    encoder = FaceEncoder()
    classifier_path = None if args.disable_classifier else Path(args.classifier_path)
    recognizer = FaceRecognizer(classifier_path=classifier_path)
    validator = ValidationRunner(recognizer=recognizer)

    if args.train:
        encoder.encode_known_faces(model=args.model)

    if args.train_classifier:
        encoder.train_classifier(
            classifier_name=args.classifier,
            classifier_path=Path(args.classifier_path),
        )

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
    
    if args.train_hyperparameter:
        recognizer_factory = RecognizerFactory(
            classifier_path=classifier_path
        )

        train_tolerance_hyperparameter(recognizer_factory, model=args.model)

    if not any(
        [args.train, args.train_classifier, args.validate, args.test, args.statistics, args.train_hyperparameter]
    ):
        parser.print_help()

