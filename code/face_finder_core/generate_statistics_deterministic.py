import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


# -----------------------------
# CORE: PURE FUNCTIONS ONLY
# -----------------------------

def calculate_accuracy(metadata_results):
    total = len(metadata_results)
    correct = 0
    per_person = {}

    for record in metadata_results:
        actual = record["actual_name"]
        pred = record["detected_name"]

        if actual not in per_person:
            per_person[actual] = {"total": 0, "correct": 0}

        per_person[actual]["total"] += 1

        if actual == pred:
            correct += 1
            per_person[actual]["correct"] += 1

    return {
        "total": total,
        "correct": correct,
        "overall_accuracy": round(correct / total * 100.0, 2) if total else 0.0,
        "per_person": {
            name: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": round(v["correct"] / v["total"] * 100.0, 2),
            }
            for name, v in sorted(per_person.items())
        },
    }


def prepare_labels_and_scores(metadata_results):
    classes = sorted(
        set(r["actual_name"] for r in metadata_results)
        | set(r["detected_name"] for r in metadata_results)
    )

    class_to_idx = {c: i for i, c in enumerate(classes)}

    y_true = [r["actual_name"] for r in metadata_results]
    y_pred = [r["detected_name"] for r in metadata_results]

    n_samples = len(metadata_results)
    n_classes = len(classes)

    y_score = np.zeros((n_samples, n_classes))

    for i, r in enumerate(metadata_results):
        idx = class_to_idx[r["detected_name"]]
        y_score[i, idx] = -float(r["confidence_distance"])

    return classes, y_true, y_pred, y_score


def calculate_classification_metrics(metadata_results):
    classes, y_true, y_pred, _ = prepare_labels_and_scores(metadata_results)

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    report = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )

    return {
        "labels": classes,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            c: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, c in enumerate(classes)
        },
        "macro_avg": report.get("macro avg", {}),
        "weighted_avg": report.get("weighted avg", {}),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def generate_statistics_from_results(results):
    return {
        "accuracy_report": calculate_accuracy(results),
        "classification_metrics": calculate_classification_metrics(results),
    }