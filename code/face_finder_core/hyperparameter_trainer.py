from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
import json
from .validation import ValidationRunner
from .generate_statistics_deterministic import generate_statistics_from_results
from .config import HYPERPARAMETERS_PATH


def _evaluate_tolerance(args):
    tolerance, model, recognizer_factory = args

    recognizer = recognizer_factory()  # fresh instance

    validator = ValidationRunner(
        recognizer=recognizer
    )

    with suppress_output():
        results = validator.run_deterministic(model=model, tolerance=tolerance)
        stats = generate_statistics_from_results(results)

    return tolerance, stats


def _parallel_search(tolerances, model, recognizer_factory):
    results = {}
    total = len(tolerances)

    with ProcessPoolExecutor() as executor:
        with tqdm(total=total, desc="Evaluating tolerances") as pbar:

            for t, stats in executor.map(
                _evaluate_tolerance,
                [(t, model, recognizer_factory) for t in tolerances]
            ):
                score = stats["classification_metrics"]["macro_avg"].get("f1-score", 0.0)
                results[t] = score

                pbar.set_postfix({"last_tol": round(t, 2), "f1": round(score, 4)})
                pbar.update(1)

    return results


def train_tolerance_hyperparameter(recognizer_factory, model):
    coarse = np.arange(0.0, 1.0, 0.1)
    print("Starting coarse search for tolerance hyperparameter...")
    coarse_results = _parallel_search(coarse, model, recognizer_factory)

    best = min(coarse_results, key=lambda t: (-coarse_results[t], t))
    print("starting fine search around best coarse tolerance: ", best)
    fine = np.arange(max(0, best - 0.1), min(best + 0.1, 1.0), 0.01)
    fine_results = _parallel_search(fine, model, recognizer_factory)

    best = min(fine_results, key=lambda t: (-fine_results[t], t))

    print(f"Best tolerance found: {best.round(2)} with F1-score: {fine_results[best]:.4f}")

    with open(HYPERPARAMETERS_PATH, "r+") as f:
        data = json.load(f)
        data["tolerance"] = best.round(2)
        f.seek(0)
        json.dump(data, f, indent=4)
    return best



import os
from contextlib import redirect_stdout, redirect_stderr


class suppress_output:
    def __enter__(self):
        self.null = open(os.devnull, "w")
        self._stdout = redirect_stdout(self.null)
        self._stderr = redirect_stderr(self.null)
        self._stdout.__enter__()
        self._stderr.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stdout.__exit__(exc_type, exc_val, exc_tb)
        self._stderr.__exit__(exc_type, exc_val, exc_tb)
        self.null.close()