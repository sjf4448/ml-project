"""Populate project training/validation folders from the LFW dataset.

The full implementation now lives in `face_finder_core.dataset` so students can
study dataset preparation in a focused module.
"""

from __future__ import annotations

from face_finder_core.dataset import KnownFacesImporter, LfwDatasetBuilder
import argparse

def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for populating training/validation folders."""
    parser = argparse.ArgumentParser(
        description="Populate training/validation folders from the LFW dataset or known faces"
    )
    parser.add_argument(
        "--lfw",
        action="store_true",
        help="Download LFW and populate training/validation folders with a deterministic split",
    )
    parser.add_argument(
        "--known",
        action="store_true",
        help="Import known faces from the 'known' folder into training/validation folders with a deterministic split",
    )
    return parser

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


def import_known_faces(
    min_faces_per_person: int = 3, validation_images_per_person: int = 1
) -> None:
    """Wrapper function for KnownFacesImport"""
    importer = KnownFacesImporter()
    importer.import_faces(
        min_faces=min_faces_per_person,
        validation_image_per=validation_images_per_person,
    )


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    
    if args.lfw:
        populate_lfw_folders(
            min_faces_per_person=20,
            resize=0.5,
            validation_images_per_person=2,
        )
    elif args.known:
        import_known_faces(min_faces_per_person=10, validation_images_per_person=5)
    else:
        parser.print_help()