import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

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
        "overall_accuracy": round(correct / total * 100.0, 2),
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
        raise FileNotFoundError(f"Metadata folder not found. Please run --validate first.")
    
    json_data_list = []
    
    for filename in os.listdir(metadata_path):
        if filename.lower().endswith(".json"):
            file_path = os.path.join(metadata_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    
                    actual_name = Path(data[0]["image_path"]).parent.name
                    
                    json_data_list.append({
                        "detected_name": data[0]["detected_name"],
                        "actual_name": actual_name,
                        "confidence_distance": data[0]["confidence_distance"]
                    })
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                
    return json_data_list

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
    bars = ax.bar(names, accuracies, color="tab:blue", edgecolor="black")
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


def generate_statistics():
    """Main point of entry to determine how the model performed"""
    results = gather_data()
    accuracy_report = calculate_accuracy(metadata_results=results)

    print(f"Total samples: {accuracy_report['total']}")
    print(f"Overall accuracy: {accuracy_report['overall_accuracy']:.2f}%")
    print("Per-person accuracy:")
    for person_name, metrics in accuracy_report["per_person"].items():
        print(
            f"  {person_name}: {metrics['correct']}/{metrics['total']} "
            f"({metrics['accuracy']:.2f}%)"
        )

    try:
        output_path = plot_accuracy(accuracy_report)
        print(f"Saved accuracy plot to: {output_path}")
    except ImportError as error:
        print(error)

    return accuracy_report

if __name__ == "__main__":
    generate_statistics()