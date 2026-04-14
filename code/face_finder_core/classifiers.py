from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


@dataclass(frozen=True)
class ClassifierSpec:
    key: str
    display_name: str


# Keep keys stable so model comparison output can be reused in CLI training flags.
_CLASSIFIER_SPECS: tuple[ClassifierSpec, ...] = (
    ClassifierSpec(key="knn", display_name="KNN(k=3)"),
    ClassifierSpec(key="logistic_regression", display_name="LogisticRegression"),
    ClassifierSpec(key="linear_svc", display_name="LinearSVC"),
    ClassifierSpec(key="random_forest", display_name="RandomForest"),
)


def available_classifier_names() -> list[str]:
    return [spec.key for spec in _CLASSIFIER_SPECS]


def classifier_display_name(name: str) -> str:
    for spec in _CLASSIFIER_SPECS:
        if spec.key == name:
            return spec.display_name
    raise ValueError(
        f"Unsupported classifier '{name}'. Available: {', '.join(available_classifier_names())}"
    )


def build_classifier(name: str):
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=3, weights="distance")
    if name == "logistic_regression":
        return LogisticRegression(max_iter=600, n_jobs=-1)
    if name == "linear_svc":
        return LinearSVC(dual="auto", max_iter=3000)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    raise ValueError(
        f"Unsupported classifier '{name}'. Available: {', '.join(available_classifier_names())}"
    )

