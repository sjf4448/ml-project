"""
ocnn.py - wrapper file for everything OCNN related. Trains, Tests, and outputs results

Note: will be integrated into recogniton pipeline at the end with the pkl file.
"""

from __future__ import annotations

import argparse
from code.face_finder_core import ocnn_embedding_db
from pathlib import Path

from face_finder_core import (
    ocnn_classifier,
    ocnn_prepare,
    ocnn_train,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OCNN — train, evaluate, and export the face embedding CNN"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── prepare ───────────────────────────────────────────────────────────────
    # runs prepare_dataset.py logic — download VGGFace2, split into folders
    prepare = subparsers.add_parser(
        "prepare", help="Download and split VGGFace2 dataset"
    )
    prepare.add_argument(
        "--max-identities",
        type=int,
        default=None,
        help="Limit number of identities (useful for prototyping, default: all)",
    )
    prepare.add_argument(
        "--min-images",
        type=int,
        default=10,
        help="Skip identities with fewer than this many images (default: 10)",
    )
    prepare.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear destination folders before copying",
    )

    # ── train ─────────────────────────────────────────────────────────────────
    train = subparsers.add_parser("train", help="Train the CNN with ArcFace loss")
    train.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Path to a checkpoint .pt file to resume training from",
    )
    train.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override NUM_EPOCHS from config",
    )
    train.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override BATCH_SIZE from config",
    )

    # ── build-db ──────────────────────────────────────────────────────────────
    # runs embedding_db.py — generates face_db.pt from the trained model
    build_db = subparsers.add_parser(
        "build-db", help="Build embedding database from trained model"
    )
    build_db.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model weights .pt file (default: checkpoints/best_model.pt)",
    )
    build_db.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Store one embedding per image instead of one per identity",
    )

    # ── evaluate ──────────────────────────────────────────────────────────────
    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate model on validation set"
    )
    evaluate.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model weights .pt file (default: checkpoints/best_model.pt)",
    )
    evaluate.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Cosine similarity threshold for recognition (default: 0.4)",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "prepare":
        ocnn_prepare.run(
            max_identities=args.max_identities,
            min_images=args.min_images,
            clear=not args.no_clear,
        )

    elif args.command == "train":
        ocnn_train.train(
            resume_from=Path(args.resume) if args.resume else None,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
        )

    elif args.command == "build-db":
        ocnn_embedding_db.build_and_save_database(
            model_path=Path(args.model_path) if args.model_path else None,
            aggregate=not args.no_aggregate,
        )

    elif args.command == "evaluate":
        ocnn_classifier.evaluate(
            model_path=Path(args.model_path) if args.model_path else None,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()
