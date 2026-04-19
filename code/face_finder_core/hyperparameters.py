from .config import HYPERPARAMETERS_PATH
import json


def get_tolerance():
    """Load the best tolerance hyperparameter from the results of hyperparameter training."""

    if not HYPERPARAMETERS_PATH.exists() or HYPERPARAMETERS_PATH.stat().st_size == 0:
        tolerance = 0.6
        return tolerance

    with HYPERPARAMETERS_PATH.open("r") as f:
        content = f.read().strip()

        if not content:
            tolerance = 0.6
            return tolerance

        data = json.loads(content)

    tolerance = data.get("tolerance")
    if tolerance is None:
        tolerance = 0.6
    if not 0.0 <= tolerance <= 1.0:
        raise ValueError(f"'tolerance' value must be between 0.0 and 1.0 in {HYPERPARAMETERS_PATH}")

    return float(tolerance)