#!/usr/bin/env python3
from __future__ import annotations

"""Train and compare multiple classifiers on face embeddings.

Workflow overview:
1) Build (or load cached) embeddings from labeled train/test folders.
2) Keep only identities present in both splits for fair evaluation.
3) Train several classifiers and compute train/test accuracy.
4) Print a compact comparison table.
"""

import argparse
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol, Sequence

import face_recognition
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from face_finder_core.classifiers import (
    available_classifier_names,
    build_classifier,
    classifier_display_name,
)
from face_finder_core.config import OUTPUT_DIR, TRAINING_DIR, VALIDATION_DIR

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class ClassifierLike(Protocol):
    """Minimal classifier interface expected by this benchmark script."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> object:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


@dataclass
class EmbeddingDataset:
    """Container for train/test embedding matrices and string labels."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


@dataclass
class ModelResult:
    """Per-model metrics reported in the final comparison table."""

    name: str
    model_key: str
    train_accuracy: float
    test_accuracy: float
    train_seconds: float


def _iter_labeled_files(base_dir: Path, max_per_class: int) -> list[tuple[str, Path]]:
    """Collect (label, file_path) pairs from a class-folder dataset.

    Expected layout: <base_dir>/<person_name>/<image files>
    """

    rows: list[tuple[str, Path]] = []
    for person_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        image_files = [
            p
            for p in sorted(person_dir.iterdir())
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]
        if max_per_class > 0:
            image_files = image_files[:max_per_class]
        rows.extend((person_dir.name, image_file) for image_file in image_files)
    return rows


def _encode_image(image_path: Path, detector_model: str) -> np.ndarray | None:
    """Return the first face embedding for an image, or None if no face is found."""

    image = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image, model=detector_model)
    if not locations:
        return None
    embeddings = face_recognition.face_encodings(image, known_face_locations=locations)
    if not embeddings:
        return None
    # Use a consistent float dtype for downstream sklearn estimators.
    return np.asarray(embeddings[0], dtype=np.float64)


def _build_split(
    base_dir: Path,
    detector_model: str,
    max_per_class: int,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one embedding split (train or test) from a labeled directory."""

    samples = _iter_labeled_files(base_dir=base_dir, max_per_class=max_per_class)
    x_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    skipped = 0

    for label, image_path in samples:
        # Skip files where no face could be embedded, but keep going.
        embedding = _encode_image(image_path=image_path, detector_model=detector_model)
        if embedding is None:
            skipped += 1
            continue
        x_rows.append(embedding)
        y_rows.append(label)

    if not x_rows:
        raise RuntimeError(
            f"No usable embeddings were produced for {split_name} at {base_dir}."
        )

    print(
        f"{split_name}: encoded {len(x_rows)} image(s), skipped {skipped}, classes={len(set(y_rows))}"
    )
    # Stack all embedding vectors into a 2D matrix: (n_samples, 128).
    return np.vstack(x_rows), np.array(y_rows)


def build_embedding_dataset(
    train_dir: Path,
    test_dir: Path,
    detector_model: str,
    max_train_per_class: int,
    max_test_per_class: int,
) -> EmbeddingDataset:
    """Build train/test embedding arrays from configured image directories."""

    x_train, y_train = _build_split(
        base_dir=train_dir,
        detector_model=detector_model,
        max_per_class=max_train_per_class,
        split_name="Train",
    )
    x_test, y_test = _build_split(
        base_dir=test_dir,
        detector_model=detector_model,
        max_per_class=max_test_per_class,
        split_name="Test",
    )

    return EmbeddingDataset(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


def _shared_classes_only(dataset: EmbeddingDataset) -> EmbeddingDataset:
    """Filter dataset to identities that appear in both train and test splits.

    This prevents evaluation on labels the model never saw during training.
    """

    train_classes = set(dataset.y_train.tolist())
    test_classes = set(dataset.y_test.tolist())
    shared = train_classes.intersection(test_classes)
    if not shared:
        raise RuntimeError("No shared classes found between train and test embeddings.")

    train_mask = np.array([y in shared for y in dataset.y_train], dtype=bool)
    test_mask = np.array([y in shared for y in dataset.y_test], dtype=bool)

    return EmbeddingDataset(
        x_train=dataset.x_train[train_mask],
        y_train=dataset.y_train[train_mask],
        x_test=dataset.x_test[test_mask],
        y_test=dataset.y_test[test_mask],
    )


def _default_models() -> list[tuple[str, ClassifierLike]]:
    """Return a small, fast model set suitable for frequent local comparisons."""
    return [
        (model_key, build_classifier(model_key))
        for model_key in available_classifier_names()
    ]


def _safe_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy while silencing expected high-class-count warnings."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="The number of unique classes is greater than 50% of the number of samples.*",
        )
        return float(accuracy_score(y_true, y_pred))


def evaluate_models(
    dataset: EmbeddingDataset,
    models: Sequence[tuple[str, ClassifierLike]],
) -> list[ModelResult]:
    """Train each classifier and collect train/test accuracy metrics."""

    # Evaluate only on shared identities so test scores are meaningful.
    dataset = _shared_classes_only(dataset)

    # Encode string class names to integer IDs for sklearn models.
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(dataset.y_train)
    y_test_encoded = label_encoder.transform(dataset.y_test)

    results: list[ModelResult] = []
    for model_key, model in models:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="The number of unique classes is greater than 50% of the number of samples.*",
            )
            # Time model training separately from prediction and metric formatting.
            started = perf_counter()
            model.fit(dataset.x_train, y_train_encoded)
            train_seconds = perf_counter() - started

            train_pred = model.predict(dataset.x_train)
            test_pred = model.predict(dataset.x_test)

        results.append(
            ModelResult(
                name=classifier_display_name(model_key),
                model_key=model_key,
                train_accuracy=_safe_accuracy(y_train_encoded, train_pred),
                test_accuracy=_safe_accuracy(y_test_encoded, test_pred),
                train_seconds=train_seconds,
            )
        )

    return results


def _print_results(results: Sequence[ModelResult]) -> None:
    """Print an ASCII table of model metrics with auto-sized columns."""

    headers = ("Model Key", "Model", "Train Accuracy", "Test Accuracy", "Train Time (s)")
    rows = [
        (
            result.model_key,
            result.name,
            f"{result.train_accuracy:.4f}",
            f"{result.test_accuracy:.4f}",
            f"{result.train_seconds:.2f}",
        )
        for result in results
    ]

    # Compute column widths from headers + row values for clean alignment.
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(char: str = "-") -> str:
        return "+" + "+".join(char * (w + 2) for w in widths) + "+"

    print("\nModel comparison results")
    print(line("="))
    print(
        "| "
        + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
        + " |"
    )
    print(line())
    for row in rows:
        print(
            "| "
            + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers)))
            + " |"
        )
    print(line("="))


def _load_or_build_dataset(args: argparse.Namespace) -> EmbeddingDataset:
    """Load cached embeddings when available, otherwise build and optionally cache."""

    cache_path = Path(args.cache_file) if args.cache_file else None

    # Fast path: reuse precomputed embeddings to avoid repeated face encoding.
    if cache_path and cache_path.exists() and not args.refresh_cache:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        print(f"Loaded embedding cache from {cache_path}")
        return EmbeddingDataset(
            x_train=payload["x_train"],
            y_train=payload["y_train"],
            x_test=payload["x_test"],
            y_test=payload["y_test"],
        )

    # Slow path: compute embeddings from image files.
    dataset = build_embedding_dataset(
        train_dir=Path(args.train_dir),
        test_dir=Path(args.test_dir),
        detector_model=args.detector_model,
        max_train_per_class=args.max_train_per_class,
        max_test_per_class=args.max_test_per_class,
    )

    # Persist embeddings so future runs can skip expensive encoding.
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(
                {
                    "x_train": dataset.x_train,
                    "y_train": dataset.y_train,
                    "x_test": dataset.x_test,
                    "y_test": dataset.y_test,
                },
                handle,
            )
        print(f"Saved embedding cache to {cache_path}")

    return dataset


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for model comparison runtime options."""

    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate several classifiers on face embeddings generated from "
            "the project's training and validation image folders."
        )
    )
    parser.add_argument("--train-dir", type=str, default=str(TRAINING_DIR))
    parser.add_argument("--test-dir", type=str, default=str(VALIDATION_DIR))
    parser.add_argument("--detector-model", choices=["hog", "cnn"], default="hog")
    parser.add_argument("--max-train-per-class", type=int, default=10)
    parser.add_argument("--max-test-per-class", type=int, default=3)
    parser.add_argument(
        "--cache-file",
        type=str,
        default=str(OUTPUT_DIR / "model_comparison_embeddings.pkl"),
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Recompute embeddings even if a cache file already exists.",
    )
    return parser


def main() -> None:
    """CLI entrypoint for embedding build/load, training, and reporting."""

    args = build_parser().parse_args()
    dataset = _load_or_build_dataset(args)

    print(
        f"Dataset sizes: train={len(dataset.y_train)}, test={len(dataset.y_test)}, "
        f"features={dataset.x_train.shape[1]}"
    )

    results = evaluate_models(dataset=dataset, models=_default_models())
    _print_results(results)


if __name__ == "__main__":
    main()

