# face_finder_core/config.py

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "face_recognition_training"
VAL_DIR = DATA_DIR / "face_recognition_validation"
TEST_DIR = DATA_DIR / "face_recognition_test"
KNOWN_DIR = DATA_DIR / "known_faces"

OUTPUT_DIR = DATA_DIR / "face_recognition_output"
ENCODINGS_PATH = OUTPUT_DIR / "encodings.pkl"
CLASSIFIER_PATH = OUTPUT_DIR / "classifier.pkl"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
CROPS_DIR = OUTPUT_DIR / "crops"
METADATA_DIR = OUTPUT_DIR / "metadata"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# ── Model ─────────────────────────────────────────────────────────────────────

BACKBONE = "resnet18"  # timm model name
PRETRAINED = True  # ImageNet init
EMBEDDING_DIM = 512
MAX_IDENTITIES = 100  # Max Identities to train on

# ── ArcFace ───────────────────────────────────────────────────────────────────

ARCFACE_SCALE = 64.0  # s — logit scale
ARCFACE_MARGIN = 0.5  # m — angular margin in radians

# ── Training ──────────────────────────────────────────────────────────────────

BATCH_SIZE = 256
NUM_EPOCHS = 30
NUM_WORKERS = 4

LR = 0.1  # SGD initial LR
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# LR schedule — step decay
LR_STEP_SIZE = 10  # drop LR every N epochs
LR_GAMMA = 0.1  # multiply LR by this on each step

WARMUP_EPOCHS = 5  # linear warmup before step decay kicks in

# ── MTCNN ─────────────────────────────────────────────────────────────────────

IMAGE_SIZE = 112  # face crop size fed to ResNet
MTCNN_MARGIN = 14  # pixels of margin around detected face
MTCNN_MIN_FACE = 20  # minimum face size in pixels to detect

# ── Augmentation ──────────────────────────────────────────────────────────────

AUGMENT_PROB = 0.5  # probability for each augmentation
COLOR_JITTER = 0.2  # brightness/contrast/saturation jitter amount

# ── Misc ──────────────────────────────────────────────────────────────────────

SEED = 42
LOG_INTERVAL = 50  # log loss every N batches
SAVE_EVERY = 5  # save checkpoint every N epochs
