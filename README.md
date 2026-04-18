# Verified Persons Facial Classification

This project is a face-recognition pipeline with two recognition modes:

1. **Distance-based matching** (original `face_recognition` behavior)
2. **Classifier-based matching** (trainable sklearn model on top of embeddings)

The code is organized so you can:
- prepare data (`code/get_faces.py`)
- build embeddings (`code/face_finder.py --train`)
- compare classifiers (`code/model_comparison.py`)
- train one chosen classifier (`code/face_finder.py --train-classifier`)
- run recognition in CLI or webcam app (`code/face_finder.py`, `code/face_finder_app.py`)

## Quickstart

From the project root, run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python code/get_faces.py
python code/face_finder.py --train --model hog
python code/model_comparison.py

# choose a model key from model_comparison output (example: linear_svc)
python code/face_finder.py --train-classifier --classifier linear_svc

python code/face_finder.py --test --file data/face_recognition_test/sam1.png
python code/face_finder_app.py --camera-index 0
```

Quick notes:
- `--model hog` is typically fastest on CPU; try `--model cnn` if you have stronger hardware.
- Lower `--tolerance` is stricter (fewer false positives, more `Unknown`).
- If you want distance-only matching, add `--disable-classifier` to CLI or app commands.

## Project Layout

```text
ml-project/
  code/
    face_finder.py               # Main CLI entry point
    face_finder_app.py           # Webcam app CLI
    get_faces.py                 # Dataset population helper script
    model_comparison.py          # Train/test comparison for available classifiers
    face_finder_core/
      cli.py
      classifiers.py
      config.py
      dataset.py
      generate_statistics.py
      recognition.py
      training.py
      validation.py
      webcam.py
  data/
    face_recognition_training/   # One subfolder per identity (train images)
    face_recognition_validation/ # One subfolder per identity (validation images)
    face_recognition_test/       # Ad-hoc test images (single-image test / webcam captures)
    known_faces/                 # Optional user-provided labeled faces
    face_recognition_output/
      encodings.pkl              # Saved embeddings + labels
      classifier.pkl             # Saved sklearn classifier + label encoder
      annotated/                 # Annotated output images
      crops/                     # Cropped face images
      metadata/                  # Per-image recognition metadata JSON
  resources/
    resources.txt
```

## Installation

Create and activate a virtual environment, then install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Or with `uv`:

```bash
uv sync
```

## Pipeline (Step by Step)

### OCNN w/ ArcFace Pipeline - Step 1: `ocnn.py`

As an alternative to the `face_recognition`/dlib embedding stack, the project includes a CNN training pipeline (`code/ocnn.py`) that fine-tunes a ResNet-18 on VGGFace2 using ArcFace metric learning. This produces a 512-d embedding model you own end-to-end.

Run the steps in order:

```bash
python code/ocnn.py prepare
python code/ocnn.py train
python code/ocnn.py build-db
python code/ocnn.py evaluate
```

#### `python code/ocnn.py prepare`

Downloads VGGFace2 via kagglehub and splits it into the training, validation, and test folders.

| Parameter | Type | Default | Meaning |
|---|---|---:|---|
| `--max-identities` | int | `MAX_IDENTITIES` in config | Limit number of identities to copy (useful for trial runs) |
| `--min-images` | int | `10` | Skip identities with fewer images than this |
| `--no-clear` | flag | off | Skip clearing destination folders before copying |

#### `python code/ocnn.py train`

Pre-aligns all images with MTCNN, then fine-tunes ResNet-18 with ArcFace loss. Saves checkpoints to `checkpoints/` and the best model to `checkpoints/best_model.pt`.

| Parameter | Type | Default | Meaning |
|---|---|---:|---|
| `--resume` | path | none | Path to a checkpoint `.pt` file to resume from |
| `--epochs` | int | `NUM_EPOCHS` in config | Override number of training epochs |
| `--batch-size` | int | `BATCH_SIZE` in config | Override batch size |

#### `python code/ocnn.py build-db`

Runs all training images through the trained model and saves the embedding database to `data/face_recognition_output/face_db.pt`. Also writes `encodings.pkl` in the format expected by the existing classifier pipeline.

| Parameter | Type | Default | Meaning |
|---|---|---:|---|
| `--model-path` | path | `checkpoints/best_model.pt` | Path to trained model weights |
| `--no-aggregate` | flag | off | Store one embedding per image instead of one per identity |

#### `python code/ocnn.py evaluate`

Runs recognition over the validation folder and prints accuracy, unknown rate, and wrong prediction rate.

| Parameter | Type | Default | Meaning |
|---|---|---:|---|
| `--model-path` | path | `checkpoints/best_model.pt` | Path to trained model weights |
| `--threshold` | float | `0.4` | Cosine similarity threshold — below this is reported as Unknown |

#### `python code/ocnn.py test`

Tests a single image against the embedding database. The person does not need to have been in the original training set — they only need their embeddings present in `face_db.pt` (added via `build-db`). Saves a cropped face, an annotated image with a bounding box, and a metadata JSON to the output folder.

```bash
python code/ocnn.py test --file data/face_recognition_test/friend1.jpg
```

| Parameter | Type | Default | Meaning |
|---|---|---:|---|
| `--file, -f` | path | required | Path to the image to test |
| `--model-path` | path | `checkpoints/best_model.pt` | Path to trained model weights |
| `--threshold` | float | `0.4` | Cosine similarity threshold — below this is reported as Unknown |
| `--show` | flag | off | Display the image after recognition |
| `--no-annotate` | flag | off | Skip saving the annotated output image |

**Testing someone not in the training set:**

```bash
# 1. Add their photos to known_faces/<name>/
# 2. Rebuild the embedding database to include them
python code/ocnn.py build-db

# 3. Test a new image of them
python code/ocnn.py test --file data/face_recognition_test/friend1.jpg --show
```

Output files written to `data/face_recognition_output/`:
- `crops/<name>_crop.jpg` — the aligned face crop passed to the model
- `annotated/<name>_annotated.jpg` — original image with bounding box and identity label
- `metadata/<name>.json` — identity, confidence, threshold, and decision

### Step 1: Build / refresh dataset folders

Run:

```bash
python code/get_faces.py
```

What it does:
- Downloads/loads LFW via `sklearn.datasets.fetch_lfw_people`
- Writes train/validation splits to:
  - `data/face_recognition_training/<person_name>/...`
  - `data/face_recognition_validation/<person_name>/...`
- Imports local user images from `data/known_faces/<name>/...` into train/validation

### Step 2: Encode training faces into embeddings

Run:

```bash
python code/face_finder.py --train
```

What it does:
- Reads all files under `data/face_recognition_training/*/*`
- Detects faces with selected detector backend (`--model hog|cnn`)
- Creates 128-d embeddings with `face_recognition.face_encodings(...)`
- Saves payload to `data/face_recognition_output/encodings.pkl`

### Step 3: Compare classifier models (optional but recommended)

Run:

```bash
python code/model_comparison.py
```

What it does:
- Builds/loads embedding matrices for train + validation sets
- Trains each available classifier key:
  - `knn`
  - `logistic_regression`
  - `linear_svc`
  - `random_forest`
- Prints a table with:
  - model key
  - model display name
  - training accuracy
  - testing accuracy
  - train time

### Step 4: Train the selected classifier for runtime use

Example (use a model key from comparison output):

```bash
python code/face_finder.py --train-classifier --classifier linear_svc
```

What it does:
- Loads `encodings.pkl`
- Trains selected sklearn model on all saved embeddings
- Saves artifact to `data/face_recognition_output/classifier.pkl`

### Step 5: Run recognition

Single-image test:

```bash
python code/face_finder.py --test --file data/face_recognition_test/sam1.png
```

Batch validation:

```bash
python code/face_finder.py --validate
```

Generate statistics (after validation metadata exists):

```bash
python code/face_finder.py --statistics
```

Webcam app:

```bash
python code/face_finder_app.py
```

## Recognition Behavior (important)

At recognition time (`--test`, `--validate`, webcam):
- The app always loads embeddings from `encodings.pkl`
- If classifier artifact exists (default `classifier.pkl`), classifier prediction is used
- The predicted identity is still gated by distance/tolerance:
  - if predicted class best distance `<= tolerance`: known label
  - otherwise: `Unknown`
- If classifier is missing/fails, logic falls back to original distance-vote matching

## CLI Parameters

### `python code/face_finder.py`

| Parameter | Type | Default | Meaning |
|---|---|---:|---|
| `--train` | flag | off | Build `encodings.pkl` from training images |
| `--train-classifier` | flag | off | Train sklearn classifier from saved embeddings |
| `--validate` | flag | off | Run recognition over validation folder |
| `--test` | flag | off | Run recognition on one image (requires `--file`) |
| `-f, --file` | path | none | Single image path for `--test` |
| `-m, --model` | `hog`/`cnn` | `hog` | Face detector backend used during encoding/recognition |
| `-t, --tolerance` | float | `0.6` | Threshold for known vs unknown decision (lower = stricter) |
| `--classifier` | key | `linear_svc` | Classifier type for `--train-classifier` |
| `--classifier-path` | path | `data/face_recognition_output/classifier.pkl` | Classifier artifact read/write path |
| `--disable-classifier` | flag | off | Force distance-only matching even if classifier exists |
| `--show` | flag | off | Display annotated image after `--test` |
| `--statistics` | flag | off | Build evaluation stats from metadata JSON files |

### `python code/face_finder_app.py`

| Parameter | Type | Default | Meaning |
|---|---|---:|---|
| `-c, --camera-index` | int | `0` | Camera device index |
| `-t, --tolerance` | float | `0.6` | Known/Unknown strictness |
| `--classifier-path` | path | `data/face_recognition_output/classifier.pkl` | Classifier artifact used during live/capture recognition |
| `--disable-classifier` | flag | off | Force distance-only mode |

### `python code/model_comparison.py`

| Parameter | Type | Default | Meaning |
|---|---|---:|---|
| `--train-dir` | path | training folder | Source folder for train embeddings |
| `--test-dir` | path | validation folder | Source folder for test embeddings |
| `--detector-model` | `hog`/`cnn` | `hog` | Detector used before embedding |
| `--max-train-per-class` | int | `10` | Cap images per class for train split (`<=0` means no cap) |
| `--max-test-per-class` | int | `3` | Cap images per class for test split (`<=0` means no cap) |
| `--cache-file` | path | `data/face_recognition_output/model_comparison_embeddings.pkl` | Embedding cache file |
| `--refresh-cache` | flag | off | Recompute embeddings even when cache exists |

## Embeddings

An embedding is a numeric vector that represents one face.

In this project:
- each training image is scanned for face locations
- each detected face is encoded into a **128-dimensional vector**
- vectors are stored with the folder label (identity)
- this serialized gallery is saved in `encodings.pkl`

During recognition, each new face is embedded the same way and compared against known embeddings.

## Models (Expanded)

### 1) Detection backends (`--model` / `--detector-model`)

- **`hog`**
  - Histogram of Oriented Gradients detector
  - CPU-friendly
  - Usually the best default for local/dev runs
- **`cnn`**
  - CNN-based face detector
  - Can be more accurate on harder images
  - Typically slower without strong hardware/GPU setup

### 2) Face embedding model

- Provided by the `face_recognition`/`dlib` stack
- Produces fixed-size 128-d embeddings for each detected face
- Used by both distance-based recognition and classifier-based recognition

### 3) Runtime identity models (classifier keys)

These are trained by `--train-classifier` and compared in `model_comparison.py`:

- **`knn` (`KNN(k=3)`)**
  - Instance-based nearest-neighbor voting in embedding space
  - Strong baseline, simple behavior
- **`logistic_regression` (`LogisticRegression`)**
  - Linear multiclass classifier with probabilistic/logit decision boundary
  - Fast and often stable on standardized embedding tasks
- **`linear_svc` (`LinearSVC`)**
  - Linear support-vector classifier
  - Typically strong on high-dimensional embeddings; default classifier key
- **`random_forest` (`RandomForest`)**
  - Ensemble of decision trees
  - Can capture nonlinear boundaries but may overfit more easily on limited samples

### 4) Distance-vote fallback model

If classifier usage is disabled or classifier artifact is unavailable:
- matching uses `compare_faces(...)` + distance scoring
- majority vote among matches selects identity
- no confident match => `Unknown`

## Notes

- Recognition quality depends heavily on image quality, lighting, and identity coverage.
- Lower `--tolerance` is stricter: fewer false positives, more `Unknown`.
- For reproducible classifier experiments, run `model_comparison.py` before `--train-classifier`.

## OCNN Explanation
We fine-tune a preexisting image model - in our case ResNet18 with the ArcFace loss function. 

### What is ArcFace?
ArcFace is a loss function specifically designed for face recognition - this fixes the typical problem with SoftMax face recognition by attempting to cluster similar 
embeddings of the same person together and other's further away. In general, it penalizes embedding with a penalty factor `m`. It also has two hyperparameters: `s` for scale, and `m`, 
the penalty.

To be more specific, ArcFace uses a hypersphere on which normalized 512-d vectors sit on. Each class gets a L2-normalized row in the weight vector of `W` with shape `(num_classes, 512)`. 
Each row represents the "center" vector of each class - the closer a vector is to that center vector(via cosine similarity), the more similar it is. During the forward pass of the training, 
an embedding `x` will have it's logit calculated via $logit_i = s * cos(\theta_i)$. $\theta_i$ represent's the angle(distance) between `x` and the i-th class center. `s` is the scale factor, 
used to push it into proper softmax/backpropogation range.

For the correct class $\theta_y$, a margin `m` is added. This reduces the logit range, forcing the model to push the correct embeddings closer together to reduce loss. Mathematically this works
since $cos(\theta + m) < cos(\theta)$

## Resource Links

See `resources/resources.txt` for source references used in the project.
