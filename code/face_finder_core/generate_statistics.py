import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
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


def calculate_accuracy(metadata_results=None):
    """
    Calculates the overall accuracy of the model.
    This function calculates individual person accuracy as well as overall model accuracy.
    """
    if metadata_results is None:
        metadata_results = gather_data()

    if not metadata_results:
        return {
            "total": 0,
            "correct": 0,
            "overall_accuracy": 0.0,
            "per_person": {},
        }

    total = 0
    correct = 0
    per_person = {}

    for record in metadata_results:
        total += 1
        actual_name = record.get("actual_name")
        detected_name = record.get("detected_name")

        if actual_name not in per_person:
            per_person[actual_name] = {"total": 0, "correct": 0}

        per_person[actual_name]["total"] += 1

        if detected_name == actual_name:
            correct += 1
            per_person[actual_name]["correct"] += 1

    return {
        "total": total,
        "correct": correct,
        "overall_accuracy": round(correct / total * 100.0, 2) if total else 0.0,
        "per_person": {
            name: {
                "total": values["total"],
                "correct": values["correct"],
                "accuracy": round(values["correct"] / values["total"] * 100.0, 2),
            }
            for name, values in sorted(per_person.items())
        },
    }


def gather_data() -> list:
    """Reads JSON output and restructures into a list"""
    metadata_path = "data/face_recognition_output/metadata/"

    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Metadata folder not found. Please run --validate first.")

    json_data_list = []

    for filename in os.listdir(metadata_path):
        if filename.lower().endswith(".json"):
            file_path = os.path.join(metadata_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                    record = data[0]
                    actual_name = Path(record["image_path"]).parent.name

                    all_distances = record.get("all_distances")
                    if isinstance(all_distances, dict):
                        all_distances = {
                            str(name): float(distance)
                            for name, distance in all_distances.items()
                        }
                    else:
                        all_distances = None

                    json_data_list.append(
                        {
                            "detected_name": record["detected_name"],
                            "actual_name": actual_name,
                            "confidence_distance": float(record["confidence_distance"])
                            if record.get("confidence_distance") is not None
                            else None,
                            "all_distances": all_distances,
                        }
                    )
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return json_data_list


def prepare_labels_and_scores(metadata_results):
    """
    Convert raw metadata results into label arrays and a score matrix.
    """
    if not metadata_results:
        return [], [], [], np.array([])

    classes = sorted(
        set(record["actual_name"] for record in metadata_results)
        | set(record["detected_name"] for record in metadata_results)
        | {
            class_name
            for record in metadata_results
            for class_name in (record.get("all_distances") or {}).keys()
        }
    )

    class_to_idx = {name: idx for idx, name in enumerate(classes)}

    y_true = [record["actual_name"] for record in metadata_results]
    y_pred = [record["detected_name"] for record in metadata_results]

    n_samples = len(metadata_results)
    n_classes = len(classes)

    y_score = np.full((n_samples, n_classes), np.nan, dtype=float)

    for i, record in enumerate(metadata_results):
        all_distances = record.get("all_distances")

        if all_distances:
            for class_name, distance in all_distances.items():
                if class_name in class_to_idx:
                    y_score[i, class_to_idx[class_name]] = -float(distance)
        else:
            pred_name = record["detected_name"]
            pred_idx = class_to_idx[pred_name]
            confidence_distance = record.get("confidence_distance")

            y_score[i, :] = 0.0

            if confidence_distance is not None:
                y_score[i, pred_idx] = -float(confidence_distance)
            else:
                y_score[i, pred_idx] = 0.0

    if np.isnan(y_score).any():
        finite_values = y_score[~np.isnan(y_score)]
        fill_value = finite_values.min() - 1.0 if finite_values.size else -1.0
        y_score = np.where(np.isnan(y_score), fill_value, y_score)

    return classes, y_true, y_pred, y_score


def calculate_classification_metrics(metadata_results=None):
    """Compute confusion matrix, precision, recall, F1, and support."""
    if metadata_results is None:
        metadata_results = gather_data()

    classes, y_true, y_pred, _ = prepare_labels_and_scores(metadata_results)

    if not y_true:
        return {
            "labels": [],
            "confusion_matrix": [],
            "per_class": {},
            "macro_avg": {},
            "weighted_avg": {},
            "accuracy": 0.0,
        }

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=classes,
        zero_division=0,
    )

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for i, label in enumerate(classes):
        per_class[label] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    return {
        "labels": classes,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "macro_avg": report.get("macro avg", {}),
        "weighted_avg": report.get("weighted avg", {}),
        "accuracy": round(float(accuracy), 4),
    }


def calculate_confidence_metrics(metadata_results=None):
    """
    Summarize how confidence_distance behaves for correct vs incorrect predictions.
    """
    if metadata_results is None:
        metadata_results = gather_data()

    if not metadata_results:
        return {}

    correct_distances = []
    incorrect_distances = []

    for record in metadata_results:
        distance = float(record["confidence_distance"])
        if record["actual_name"] == record["detected_name"]:
            correct_distances.append(distance)
        else:
            incorrect_distances.append(distance)

    def summarize(values):
        if not values:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
                "std": None,
            }
        arr = np.array(values, dtype=float)
        return {
            "count": int(arr.size),
            "mean": round(float(arr.mean()), 4),
            "median": round(float(np.median(arr)), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "std": round(float(arr.std(ddof=0)), 4),
        }

    return {
        "correct_predictions": summarize(correct_distances),
        "incorrect_predictions": summarize(incorrect_distances),
    }


def calculate_multiclass_roc(metadata_results=None):
    """
    Compute one-vs-rest ROC curves and AUC for each class.
    """
    if metadata_results is None:
        metadata_results = gather_data()

    classes, y_true, _, y_score = prepare_labels_and_scores(metadata_results)

    if not y_true or len(classes) < 2:
        return {"labels": classes, "roc": {}, "micro_auc": None}

    y_true_bin = label_binarize(y_true, classes=classes)

    roc_data = {}

    for i, class_name in enumerate(classes):
        # Need both positive and negative samples
        if len(np.unique(y_true_bin[:, i])) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        roc_data[class_name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(roc_auc),
        }

    micro_auc = None
    if y_true_bin.shape[1] > 1:
        try:
            fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
            micro_auc = float(auc(fpr_micro, tpr_micro))
        except ValueError:
            micro_auc = None

    return {
        "labels": classes,
        "roc": roc_data,
        "micro_auc": micro_auc,
    }


def plot_accuracy(accuracy_report, output_path=None):
    """Render and save an accuracy bar chart for the validation results."""
    if not accuracy_report["per_person"]:
        raise ValueError("No per-person accuracy data available to plot.")

    if output_path is None:
        output_path = Path("data") / "face_recognition_output" / "accuracy_plot.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(accuracy_report["per_person"].keys())
    accuracies = [metrics["accuracy"] for metrics in accuracy_report["per_person"].values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, accuracies, edgecolor="black")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Person")
    ax.set_title("Per-person recognition accuracy")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(
            f"{value:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_confusion_matrix(metrics_report, output_path=None):
    """Render and save a confusion matrix heatmap."""
    labels = metrics_report["labels"]
    cm = np.array(metrics_report["confusion_matrix"])

    if cm.size == 0:
        raise ValueError("No confusion matrix data available to plot.")

    if output_path is None:
        output_path = Path("data") / "face_recognition_output" / "confusion_matrix.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_confidence_histogram(metadata_results=None, output_path=None):
    """Plot distance distributions for correct vs incorrect predictions."""
    if metadata_results is None:
        metadata_results = gather_data()

    if output_path is None:
        output_path = Path("data") / "face_recognition_output" / "confidence_histogram.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct_distances = []
    incorrect_distances = []

    for record in metadata_results:
        distance = float(record["confidence_distance"])
        if record["actual_name"] == record["detected_name"]:
            correct_distances.append(distance)
        else:
            incorrect_distances.append(distance)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(correct_distances, bins=20, alpha=0.6, label="Correct")
    ax.hist(incorrect_distances, bins=20, alpha=0.6, label="Incorrect")
    ax.set_title("Confidence Distance Distribution")
    ax.set_xlabel("Confidence Distance (lower is better)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_roc_curves(roc_report, output_path=None):
    """Plot one-vs-rest ROC curves for each class."""
    if output_path is None:
        output_path = Path("data") / "face_recognition_output" / "roc_curves.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not roc_report["roc"]:
        raise ValueError("No ROC data available to plot.")

    fig, ax = plt.subplots(figsize=(10, 8))

    for class_name, values in roc_report["roc"].items():
        ax.plot(values["fpr"], values["tpr"], label=f"{class_name} (AUC={values['auc']:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def generate_statistics():
    """Main point of entry to determine how the model performed"""
    results = gather_data()

    accuracy_report = calculate_accuracy(metadata_results=results)
    classification_metrics = calculate_classification_metrics(metadata_results=results)
    confidence_metrics = calculate_confidence_metrics(metadata_results=results)
    roc_report = calculate_multiclass_roc(metadata_results=results)

    print(f"Total samples: {accuracy_report['total']}")
    print(f"Overall accuracy: {accuracy_report['overall_accuracy']:.2f}%")
    print()

    print("Per-person accuracy:")
    for person_name, metrics in accuracy_report["per_person"].items():
        print(
            f"  {person_name}: {metrics['correct']}/{metrics['total']} "
            f"({metrics['accuracy']:.2f}%)"
        )
    print()

    print("Precision / Recall / F1 by class:")
    for class_name, metrics in classification_metrics["per_class"].items():
        print(
            f"  {class_name}: "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"support={metrics['support']}"
        )
    print()

    print("Macro average:")
    print(
        f"  precision={classification_metrics['macro_avg'].get('precision', 0):.4f}, "
        f"recall={classification_metrics['macro_avg'].get('recall', 0):.4f}, "
        f"f1={classification_metrics['macro_avg'].get('f1-score', 0):.4f}"
    )
    print()

    print("Confidence-distance summary:")
    print(f"  Correct predictions:   {confidence_metrics['correct_predictions']}")
    print(f"  Incorrect predictions: {confidence_metrics['incorrect_predictions']}")
    print()

    if roc_report["micro_auc"] is not None:
        print(f"Micro-average ROC AUC: {roc_report['micro_auc']:.4f}")

    print("Per-class ROC AUC:")
    for class_name, values in roc_report["roc"].items():
        print(f"  {class_name}: AUC={values['auc']:.4f}")
    print()

    try:
        accuracy_path = plot_accuracy(accuracy_report)
        print(f"Saved accuracy plot to: {accuracy_path}")
    except Exception as error:
        print(f"Could not save accuracy plot: {error}")

    try:
        confusion_path = plot_confusion_matrix(classification_metrics)
        print(f"Saved confusion matrix plot to: {confusion_path}")
    except Exception as error:
        print(f"Could not save confusion matrix plot: {error}")

    try:
        confidence_path = plot_confidence_histogram(results)
        print(f"Saved confidence histogram to: {confidence_path}")
    except Exception as error:
        print(f"Could not save confidence histogram: {error}")

    try:
        roc_path = plot_roc_curves(roc_report)
        print(f"Saved ROC curves plot to: {roc_path}")
    except Exception as error:
        print(f"Could not save ROC curves: {error}")

    return {
        "accuracy_report": accuracy_report,
        "classification_metrics": classification_metrics,
        "confidence_metrics": confidence_metrics,
        "roc_report": roc_report,
    }


if __name__ == "__main__":
    generate_statistics()