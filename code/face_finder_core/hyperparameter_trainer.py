import json

from matplotlib.pyplot import get
import numpy as np
from numpy.compat import Path
from sklearn.metrics import precision_recall_fscore_support

from .recognition import FaceRecognizer

from .validation import ValidationRunner

from .generate_statistics import generate_statistics
"""
Based on data saved to `data/face_recognition_output/classifier.pkl`, train the tolerance hyperparameter by evaluating recognition performance on the validation set. Starting with a first pass of tolerance=0.0, iterating up to 1.0 in incrememnts of 0.1.
Then determine the optimal range to search more finely and return the best tolerance value based on F1 score. 

    validation_data = {
        "accuracy_report": accuracy_report,
        "classification_metrics": classification_metrics,
        "confidence_metrics": confidence_metrics,
        "roc_report": roc_report, 
    }
"""
def train_tolerance_hyperparameter(validator:ValidationRunner, model):
    """
    Iterate over a range of tolerance values, run validation, compute metrics. If The results are lower than the previous best, then iterate more finely around the previous best value. Return the best tolerance value based on F1 score.
    """
    path = Path("data/face_recognition_output/validation_summary.json")
    last = 0
    tolerance = 0.0
    for tolerance in np.arange(0.0, 1.1, 0.1):
        print(f"\n\n\nTesting tolerance: {tolerance.round(2)}\n\n\n")
        validator.run(model=model, tolerance=tolerance)
        generate_statistics()
        with open(path, "r") as f:
            validation_data = json.load(f)
            f1_score = validation_data["classification_metrics"]["macro_avg"].get("f1-score", 0.0)
            if f1_score < last:
                break
            last = f1_score
    # Iterate more finely around the previous best value
    print(f"\n\n\nBest tolerance so far: {tolerance.round(2)} with F1 score: {last}\nRefining Search...\n\n\n")
    best_tolerance = tolerance - 0.2 if tolerance > 0.2 else 0.0
    for tolerance in np.arange(best_tolerance, best_tolerance + 0.3, 0.05):
        print(f"\n\n\nTesting tolerance: {tolerance.round(2)}\n\n\n")
        validator.run(model=model, tolerance=tolerance)
        generate_statistics()
        with open(path, "r") as f:
            validation_data = json.load(f)
            f1_score = validation_data["classification_metrics"]["macro_avg"].get("f1-score", 0.0)
            if f1_score > last:
                last = f1_score
                best_tolerance = tolerance
            else:
                break
    print(f"Best tolerance: {best_tolerance} with F1 score: {last}")
    return best_tolerance
