"""Populate project training/validation folders from the LFW dataset.

The full implementation now lives in `face_finder_core.dataset` so students can
study dataset preparation in a focused module.
"""

from __future__ import annotations

from face_finder_core.dataset import LfwDatasetBuilder


def populate_lfw_folders(
    min_faces_per_person: int = 20,
    resize: float = 0.5,
    validation_images_per_person: int = 2,
) -> None:
    """Compatibility function that delegates to `LfwDatasetBuilder.populate`."""
    builder = LfwDatasetBuilder()
    builder.populate(
        min_faces_per_person=min_faces_per_person,
        resize=resize,
        validation_images_per_person=validation_images_per_person,
    )


if __name__ == "__main__":
    populate_lfw_folders(
        min_faces_per_person=20,
        resize=0.5,
        validation_images_per_person=2,
    )