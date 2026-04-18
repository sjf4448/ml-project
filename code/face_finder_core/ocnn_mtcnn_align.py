# prealign_faces.py
import shutil
from pathlib import Path

from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

from .ocnn_config import TEST_DIR, TRAIN_DIR, VAL_DIR

DIRS = [TRAIN_DIR, VAL_DIR, TEST_DIR]

IMAGE_SIZE = 112
MARGIN = 14


def run():
    mtcnn = MTCNN(
        image_size=IMAGE_SIZE, margin=MARGIN, keep_all=False, post_process=False
    )

    for root in DIRS:
        images = list(root.rglob("*.jpg")) + list(root.rglob("*.png"))
        failed = 0
        for img_path in tqdm(images, desc=str(root.name)):
            pil = Image.open(img_path).convert("RGB")
            face = mtcnn(pil)
            if face is None:
                img_path.unlink()  # remove images with no detectable face
                failed += 1
                continue
            # overwrite in place with the aligned crop
            aligned = Image.fromarray(face.permute(1, 2, 0).numpy().astype("uint8"))
            aligned.save(img_path)
        print(f"  {failed} images removed (no face detected)")
