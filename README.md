# Verified persons facial classification

## `code/face_recognition.py`

This module implements the **face detection and baseline recognition pipeline** used in the project. It is adapted from the Real Python face recognition tutorial and modified to match the project's directory structure and workflow.

The primary purpose of this file is to:

1. **Detect faces in images**
2. **Generate facial encodings for known individuals**
3. **Recognize known faces in new images**
4. **Export cropped faces and metadata for downstream models**

This script serves as the **preprocessing stage** for the broader facial classification system.

---

### Directory Structure

The script assumes the following project layout:
data/
├── face_recognition_training/
│ ├── person_a/
│ │ ├── img1.jpg
│ │ └── img2.jpg
│ └── person_b/
│ ├── img1.jpg
│ └── img2.jpg
│
├── face_recognition_validation/
│ ├── image1.jpg
│ └── image2.jpg
│
└── face_recognition_output/
├── annotated/ # images with bounding boxes and labels
├── crops/ # cropped faces extracted from images
└── metadata/ # JSON detection metadata

---

### Key Features

#### Training (`--train`)
Encodes known faces from the training dataset and saves them to:
data/face_recognition_output/encodings.pkl

Each subdirectory name in `face_recognition_training` is treated as the **identity label**.

Example:
face_recognition_training/
alice/
bob/

---

#### Face Detection and Recognition (`--test`)
Detects faces in an input image and attempts to match them against the known encodings.

For each detected face the script:

- identifies the best matching known person (if any)
- draws a bounding box around the face
- saves a cropped face image
- records metadata about the detection

Outputs include:
annotated/<image>annotated.png
crops/<image>face<n><label>.png
metadata/<image>_detections.json

---

#### Validation (`--validate`)
Runs recognition on every image inside:
data/face_recognition_validation/

This is useful for evaluating detection performance before integrating the classifier.

---

### Example Usage

Train encodings:

```bash
python code/face_recognition.py --train
```
Run validation on a dataset:
```bash
python code/face_recognition.py --validate
```
Test on a single image:
```bash
python code/face_recognition.py --test --file path/to/image.jpg
```
Display results visually:
```bash
python code/face_recognition.py --test --file image.jpg --show
```
### Role in the Overall Project
This module acts as the **face localization and preprocessing stage** of the facial classification system.
Pipeline:

Image
   ↓
Face Detection (this script)
   ↓
Face Cropping
   ↓
Classifier Model
   ↓
Final Identity Prediction

Even if a separate deep learning classifier is used later, this script provides:
reliable face localization
standardized face crops
metadata for evaluation
a baseline recognition system for comparison