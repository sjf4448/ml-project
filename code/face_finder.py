from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import face_recognition
from PIL import Image, ImageDraw

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "data" / "face_recognition_training"
VALIDATION_DIR = PROJECT_ROOT / "data" / "face_recognition_validation"
OUTPUT_DIR = PROJECT_ROOT / "data" / "face_recognition_output"
ENCODINGS_PATH = OUTPUT_DIR / "encodings.pkl"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
CROPS_DIR = OUTPUT_DIR / "crops"
METADATA_DIR = OUTPUT_DIR / "metadata"

BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"


@dataclass
class DetectionResult:
    image_path: str
    detected_name: str
    top: int
    right: int
    bottom: int
    left: int
    confidence_distance: float | None
    crop_path: str | None


def ensure_directories() -> None:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def encode_known_faces(
    model: str = "hog",
    encodings_location: Path = ENCODINGS_PATH,
) -> None:
    """Train the recognition stage by encoding one or more faces per image.

    Directory layout:
        data/face_recognition_training/
            person_a/
                image1.jpg
                image2.jpg
            person_b/
                image1.jpg

    Each subdirectory name becomes that person's label.
    """
    ensure_directories()

    names: list[str] = []
    encodings: list[Any] = []
    processed_files = 0
    skipped_files = 0

    for filepath in TRAINING_DIR.glob("*/*"):
        if not filepath.is_file():
            continue

        processed_files += 1
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if not face_encodings:
            skipped_files += 1
            print(f"[WARN] No faces found in training image: {filepath}")
            continue

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    if not encodings:
        raise RuntimeError(
            "No face encodings were created. Check that your training folders contain valid images with visible faces."
        )

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open("wb") as f:
        pickle.dump(name_encodings, f)

    print(
        f"Saved {len(encodings)} encodings from {processed_files - skipped_files} usable training images to {encodings_location}"
    )


def _recognize_face(
    unknown_encoding: Any,
    loaded_encodings: dict[str, list[Any]],
    tolerance: float = 0.6,
) -> tuple[str | None, float | None]:
    """Return the best name vote and the best face distance.

    Lower distance means a closer match. If nothing is within tolerance,
    returns (None, best_distance_or_None).
    """
    known_encodings = loaded_encodings["encodings"]
    known_names = loaded_encodings["names"]

    if not known_encodings:
        return None, None

    boolean_matches = face_recognition.compare_faces(
        known_encodings,
        unknown_encoding,
        tolerance=tolerance,
    )
    face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)

    votes = Counter(
        name for match, name in zip(boolean_matches, known_names) if match
    )

    best_distance = float(face_distances.min()) if len(face_distances) else None

    if votes:
        return votes.most_common(1)[0][0], best_distance

    return None, best_distance


def _display_face(draw: ImageDraw.ImageDraw, bounding_box: tuple[int, int, int, int], name: str) -> None:
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR, width=3)
    text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill=BOUNDING_BOX_COLOR,
        outline=BOUNDING_BOX_COLOR,
    )
    draw.text((text_left, text_top), name, fill=TEXT_COLOR)


def _save_face_crop(
    pillow_image: Image.Image,
    bounding_box: tuple[int, int, int, int],
    image_stem: str,
    face_index: int,
    name: str,
) -> str:
    top, right, bottom, left = bounding_box
    crop = pillow_image.crop((left, top, right, bottom))
    safe_name = name.replace(" ", "_")
    crop_path = CROPS_DIR / f"{image_stem}_face_{face_index}_{safe_name}.png"
    crop.save(crop_path)
    return str(crop_path)


def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = ENCODINGS_PATH,
    tolerance: float = 0.6,
    save_output: bool = True,
    show_image: bool = False,
) -> list[DetectionResult]:
    """Detect, identify, annotate, and optionally save outputs for one image.

    This is the key bridge between face localization and downstream classification.
    Even if you later replace the recognizer with your own classifier, the bounding
    boxes and crops produced here can feed that model.
    """
    ensure_directories()

    if not encodings_location.exists():
        raise FileNotFoundError(
            f"Encodings file not found at {encodings_location}. Run with --train first."
        )

    with encodings_location.open("rb") as f:
        loaded_encodings = pickle.load(f)

    input_image_path = Path(image_location)
    input_image = face_recognition.load_image_file(input_image_path)
    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(
        input_image,
        input_face_locations,
    )

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    results: list[DetectionResult] = []

    for idx, (bounding_box, unknown_encoding) in enumerate(
        zip(input_face_locations, input_face_encodings),
        start=1,
    ):
        name, best_distance = _recognize_face(
            unknown_encoding,
            loaded_encodings,
            tolerance=tolerance,
        )
        if not name:
            name = "Unknown"

        _display_face(draw, bounding_box, name)

        crop_path = None
        if save_output:
            crop_path = _save_face_crop(
                pillow_image=pillow_image,
                bounding_box=bounding_box,
                image_stem=input_image_path.stem,
                face_index=idx,
                name=name,
            )

        result = DetectionResult(
            image_path=str(input_image_path),
            detected_name=name,
            top=bounding_box[0],
            right=bounding_box[1],
            bottom=bounding_box[2],
            left=bounding_box[3],
            confidence_distance=best_distance,
            crop_path=crop_path,
        )
        results.append(result)
        print(
            f"{name}: box={bounding_box}, best_distance={best_distance}"
        )

    del draw

    if save_output:
        annotated_path = ANNOTATED_DIR / f"{input_image_path.stem}_annotated.png"
        metadata_path = METADATA_DIR / f"{input_image_path.stem}_detections.json"
        pillow_image.save(annotated_path)
        metadata_path.write_text(
            json.dumps([asdict(r) for r in results], indent=2),
            encoding="utf-8",
        )
        print(f"Annotated image saved to {annotated_path}")
        print(f"Detection metadata saved to {metadata_path}")

    if show_image:
        pillow_image.show()

    return results


def validate(model: str = "hog", tolerance: float = 0.6) -> None:
    ensure_directories()

    validation_files = [
        filepath for filepath in VALIDATION_DIR.rglob("*") if filepath.is_file()
    ]

    if not validation_files:
        print(f"No validation images found in {VALIDATION_DIR}")
        return

    for filepath in validation_files:
        print(f"\n[VALIDATE] {filepath}")
        recognize_faces(
            image_location=str(filepath),
            model=model,
            tolerance=tolerance,
            save_output=True,
            show_image=False,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Face detection and recognition pipeline for known-person classification"
    )
    parser.add_argument("--train", action="store_true", help="Encode labeled training faces")
    parser.add_argument("--validate", action="store_true", help="Run recognition on every image in validation")
    parser.add_argument("--test", action="store_true", help="Run recognition on one image")
    parser.add_argument(
        "-f",
        "--file",
        dest="file",
        help="Path to a single image for testing",
    )
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.train:
        encode_known_faces(model=args.model)

    if args.validate:
        validate(model=args.model, tolerance=args.tolerance)

    if args.test:
        if not args.file:
            parser.error("--test requires --file PATH")
        recognize_faces(
            image_location=args.file,
            model=args.model,
            tolerance=args.tolerance,
            save_output=True,
            show_image=args.show,
        )

    if not any([args.train, args.validate, args.test]):
        parser.print_help()


if __name__ == "__main__":
    main()
