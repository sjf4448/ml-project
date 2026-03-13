from __future__ import annotations

import sys
import time
import subprocess
from pathlib import Path

import cv2

import face_recognition

from face_finder import ANNOTATED_DIR, ensure_directories, recognize_faces


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = PROJECT_ROOT / "data" / "face_recognition_test"
WINDOW_NAME = "Face Finder App"


def open_file(path: Path) -> None:
    """Open a file with the system default image viewer."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform.startswith("win"):
            subprocess.run(["cmd", "/c", "start", "", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as exc:
        print(f"[WARN] Could not automatically open file: {exc}")


def capture_photo() -> Path | None:
    """Open the webcam and save one captured image to the test directory.

    Controls:
    - Press SPACE to take a picture
    - Press Q to quit without saving
    """
    ensure_directories()
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[ERROR] Could not open webcam.")
        return None

    print("Webcam opened.")
    print("Press SPACE to capture a photo.")
    print("Press Q to quit.")

    saved_path: Path | None = None
    frame_count = 0
    process_every_n_frames = 2
    scale = 0.25
    cached_face_locations: list[tuple[int, int, int, int]] = []

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("[ERROR] Failed to read a frame from the webcam.")
                break

            frame_count += 1

            if frame_count % process_every_n_frames == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                small_face_locations = face_recognition.face_locations(
                    rgb_small_frame,
                    model="hog",
                )

                cached_face_locations = [
                    (
                        int(top / scale),
                        int(right / scale),
                        int(bottom / scale),
                        int(left / scale),
                    )
                    for top, right, bottom, left in small_face_locations
                ]

            display_frame = frame.copy()

            for top, right, bottom, left in cached_face_locations:
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.putText(
                display_frame,
                f"Faces detected: {len(cached_face_locations)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                "SPACE: capture   Q: quit",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 32:  # SPACE
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                saved_path = TEST_DIR / f"webcam_capture_{timestamp}.png"
                cv2.imwrite(str(saved_path), frame)
                print(f"Saved image to {saved_path}")
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return saved_path


def run_test(image_path: Path) -> None:
    """Run the face-finder pipeline on the saved image and open the annotation."""
    print(f"Running face detection and recognition on {image_path}...")
    results = recognize_faces(
        image_location=str(image_path),
        model="hog",
        save_output=True,
        show_image=False,
    )

    annotated_path = ANNOTATED_DIR / f"{image_path.stem}_annotated.png"

    if results:
        print(f"Detected {len(results)} face(s).")
    else:
        print("No faces were detected in the image.")

    if annotated_path.exists():
        print(f"Opening annotated image: {annotated_path}")
        open_file(annotated_path)
    else:
        print("[WARN] Annotated image was not found.")



def main() -> None:
    captured_path = capture_photo()
    if captured_path is None:
        print("No image captured.")
        return

    run_test(captured_path)


if __name__ == "__main__":
    main()
