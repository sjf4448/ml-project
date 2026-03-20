# Verified Persons Facial Classification

This project is an educational face-recognition pipeline built around three stages:

1. Build a labeled dataset (`get_faces.py`)
2. Encode known people (`face_finder.py --train`)
3. Detect and recognize faces in new images (`face_finder.py --test` / `--validate`)

It also includes a small webcam app (`face_finder_app.py`) for interactive testing.

## Project Layout

```text
ml-project/
  code/
    face_finder.py              # Main CLI (training, test, validation)
    face_finder_app.py          # Webcam capture + recognition app
    get_faces.py                # LFW dataset preparation script
    face_finder_core/           
      __init__.py
      cli.py
      config.py
      dataset.py
      recognition.py
      training.py
      validation.py
      webcam.py
  data/
    face_recognition_training/  # Known identities (one folder per person)
    face_recognition_validation/# Validation set
    face_recognition_test/      # Ad-hoc test images / webcam captures
    face_recognition_output/
      encodings.pkl             # Serialized known-face embeddings
      annotated/                # Images with boxes + labels
      crops/                    # Cropped face images
      metadata/                 # JSON for each processed image
  resources/
    resources.txt               # Source/tutorial links used in the project
```

## How the Program Works

### 1) Dataset preparation (`code/get_faces.py`)

- Uses `scikit-learn` `fetch_lfw_people` to download LFW faces.
- Writes images into this project structure:
  - `data/face_recognition_training/<person_name>/...`
  - `data/face_recognition_validation/<person_name>/...`
- Splits per identity: first `N` images for validation, remainder for training.
- User provided faces are placed into `data/known_faces/<name>`
  - name should be in format: `first last`
- will also be written into the above project structure

Core class: `face_finder_core.dataset.LfwDatasetBuilder`

### 2) Training / encoding (`code/face_finder.py --train`)

- Reads all images in `data/face_recognition_training/*/*`.
- Detects faces and computes 128-d embeddings with `face_recognition`.
- Saves known names + embeddings to `data/face_recognition_output/encodings.pkl`.

Core class: `face_finder_core.training.FaceEncoder`

### 3) Recognition (`code/face_finder.py --test`)

- Loads the saved encodings file.
- Detects faces in a target image.
- Compares each detected face embedding to known embeddings.
- Assigns the best matching identity (or `Unknown`).
- Saves:
  - Annotated image (box + label)
  - Cropped face image(s)
  - JSON metadata with box coordinates and distance score

Core class: `face_finder_core.recognition.FaceRecognizer`

### 4) Batch validation (`code/face_finder.py --validate`)

- Runs recognition on every file in `data/face_recognition_validation/`.
- Produces the same output artifacts as single-image testing.

Core class: `face_finder_core.validation.ValidationRunner`

### 5) Webcam app (`code/face_finder_app.py`)

- Opens webcam preview.
- Shows live face boxes with identity coloring:
  - Green: recognized as a known person
  - Red: not matched (Unknown)
- Press `SPACE` to capture, then runs recognition on that frame.
- Opens the resulting annotated image.

Core class: `face_finder_core.webcam.WebcamCaptureSession`

## Installation

Create and activate a virtual environment, then install dependencies. `uv` has also set up, so `uv sync` will also work.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**OR** 

```bash
uv sync
```

## Usage

Run commands from the project root (`ml-project/`).

### Prepare dataset (optional if you already have images)

```bash
python code/get_faces.py
```

### Train known encodings

```bash
python code/face_finder.py --train
```

### Test a single image

```bash
python code/face_finder.py --test --file data/face_recognition_test/sam1.png
```

### Test and display annotated image

```bash
python code/face_finder.py --test --file data/face_recognition_test/sam1.png --show
```

### Run batch validation

```bash
python code/face_finder.py --validate
```

### Launch webcam app

```bash
python code/face_finder_app.py
```

### Launch webcam app with a specific camera index (macOS tip)

```bash
python code/face_finder_app.py --camera-index 1
```

If Continuity Camera opens by default, try `--camera-index 1` or `--camera-index 2`.

### Launch webcam app with custom recognition tolerance

```bash
python code/face_finder_app.py --tolerance 0.5
```

Tip: lower tolerance is stricter, so fewer faces are marked as known.

## Output Files

After recognition, look in `data/face_recognition_output/`:

- `annotated/<image>_annotated.png`
- `crops/<image>_face_<n>_<label>.png`
- `metadata/<image>_detections.json`

`metadata` JSON records:
- predicted name
- bounding box (`top`, `right`, `bottom`, `left`)
- distance score (`confidence_distance`, lower means closer)
- crop path

## Notes

- Recognition quality depends heavily on training image quality and variety.
- Lower `--tolerance` gives stricter matching (fewer false positives, more unknowns).
- `hog` model is CPU-friendly; `cnn` can be more accurate but needs stronger hardware.

## Resource Links

See `resources/resources.txt` for source references used to build this project.

## Embeddings

In this project, an **embedding** is a numeric vector representation of one detected face.

- During `--train`, each image in `data/face_recognition_training/*/*` is loaded.
- Faces are located with `face_recognition.face_locations(...)`.
- For each located face, `face_recognition.face_encodings(...)` generates a 128-d embedding.
- Each embedding is stored alongside its folder label (person name).
- All known labels + embeddings are serialized to `data/face_recognition_output/encodings.pkl`.

At recognition time (`--test`, `--validate`, webcam), the pipeline computes embeddings for detected faces in the new image/frame and compares them against the stored embeddings from `encodings.pkl`.


## Models

This project uses pretrained models through the `face_recognition`/`dlib` stack

- `--model hog` or `--model cnn` selects the **face detection backend**.
  - `hog`: CPU-friendly, less accurate, works well for weaker hardware.
  - `cnn`: GPU-accelerated, can be more accurate but requires stronger hardware.
- Face embeddings are generated by the pretrained face encoder used by `face_recognition`.
- Identity assignment is done by comparing embedding distances to known embeddings:
  - `compare_faces(...)` applies the tolerance threshold.
  - `face_distance(...)` provides closeness scores (lower is better).
  - A majority vote across matched embeddings picks the final label; otherwise it returns `Unknown`.
