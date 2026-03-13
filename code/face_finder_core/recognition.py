from __future__ import annotations

import json
import pickle
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import face_recognition
from PIL import Image, ImageDraw

from .config import (
    ANNOTATED_DIR,
    BOUNDING_BOX_COLOR,
    CROPS_DIR,
    ENCODINGS_PATH,
    METADATA_DIR,
    TEXT_COLOR,
    ensure_directories,
)


@dataclass
class DetectionResult:
    """Serializable details for one detected face."""

    image_path: str
    detected_name: str
    top: int
    right: int
    bottom: int
    left: int
    confidence_distance: float | None
    crop_path: str | None


class FaceRecognizer:
    """Runs face detection + identity matching for one input image."""

    def __init__(self, encodings_location: Path = ENCODINGS_PATH):
        self.encodings_location = encodings_location

    def _load_known_encodings(self) -> dict[str, list[Any]]:
        if not self.encodings_location.exists():
            raise FileNotFoundError(
                f"Encodings file not found at {self.encodings_location}. Run training first."
            )

        with self.encodings_location.open("rb") as handle:
            return pickle.load(handle)

    @staticmethod
    def _recognize_face(
        unknown_encoding: Any,
        loaded_encodings: dict[str, list[Any]],
        tolerance: float,
    ) -> tuple[str | None, float | None]:
        """Return the top voted name and best distance for one face embedding."""
        known_encodings = loaded_encodings["encodings"]
        known_names = loaded_encodings["names"]

        if not known_encodings:
            return None, None

        matches = face_recognition.compare_faces(
            known_encodings,
            unknown_encoding,
            tolerance=tolerance,
        )
        face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)

        votes = Counter(name for match, name in zip(matches, known_names) if match)
        best_distance = float(face_distances.min()) if len(face_distances) else None

        if votes:
            return votes.most_common(1)[0][0], best_distance

        return None, best_distance

    @staticmethod
    def _display_face(
        draw: ImageDraw.ImageDraw,
        bounding_box: tuple[int, int, int, int],
        label: str,
    ) -> None:
        """Draw a rectangle and text label for one face."""
        top, right, bottom, left = bounding_box
        draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR, width=3)
        text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), label)
        draw.rectangle(
            ((text_left, text_top), (text_right, text_bottom)),
            fill=BOUNDING_BOX_COLOR,
            outline=BOUNDING_BOX_COLOR,
        )
        draw.text((text_left, text_top), label, fill=TEXT_COLOR)

    @staticmethod
    def _save_face_crop(
        pillow_image: Image.Image,
        bounding_box: tuple[int, int, int, int],
        image_stem: str,
        face_index: int,
        label: str,
    ) -> str:
        """Persist a cropped face image for downstream models or manual inspection."""
        top, right, bottom, left = bounding_box
        crop = pillow_image.crop((left, top, right, bottom))
        safe_label = label.replace(" ", "_")
        crop_path = CROPS_DIR / f"{image_stem}_face_{face_index}_{safe_label}.png"
        crop.save(crop_path)
        return str(crop_path)

    def recognize_faces(
        self,
        image_location: str,
        model: str = "hog",
        tolerance: float = 0.6,
        save_output: bool = True,
        show_image: bool = False,
    ) -> list[DetectionResult]:
        """Recognize all faces in a single image and optionally save artifacts."""
        ensure_directories()
        loaded_encodings = self._load_known_encodings()

        input_image_path = Path(image_location)
        input_image = face_recognition.load_image_file(input_image_path)
        input_face_locations = face_recognition.face_locations(input_image, model=model)
        input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

        pillow_image = Image.fromarray(input_image)
        draw = ImageDraw.Draw(pillow_image)
        results: list[DetectionResult] = []

        for index, (bounding_box, unknown_encoding) in enumerate(
            zip(input_face_locations, input_face_encodings),
            start=1,
        ):
            name, best_distance = self._recognize_face(
                unknown_encoding=unknown_encoding,
                loaded_encodings=loaded_encodings,
                tolerance=tolerance,
            )
            if not name:
                name = "Unknown"

            self._display_face(draw=draw, bounding_box=bounding_box, label=name)

            crop_path = None
            if save_output:
                crop_path = self._save_face_crop(
                    pillow_image=pillow_image,
                    bounding_box=bounding_box,
                    image_stem=input_image_path.stem,
                    face_index=index,
                    label=name,
                )

            results.append(
                DetectionResult(
                    image_path=str(input_image_path),
                    detected_name=name,
                    top=bounding_box[0],
                    right=bounding_box[1],
                    bottom=bounding_box[2],
                    left=bounding_box[3],
                    confidence_distance=best_distance,
                    crop_path=crop_path,
                )
            )
            print(f"{name}: box={bounding_box}, best_distance={best_distance}")

        del draw

        if save_output:
            annotated_path = ANNOTATED_DIR / f"{input_image_path.stem}_annotated.png"
            metadata_path = METADATA_DIR / f"{input_image_path.stem}_detections.json"
            pillow_image.save(annotated_path)
            metadata_path.write_text(
                json.dumps([asdict(result) for result in results], indent=2),
                encoding="utf-8",
            )
            print(f"Annotated image saved to {annotated_path}")
            print(f"Detection metadata saved to {metadata_path}")

        if show_image:
            pillow_image.show()

        return results

