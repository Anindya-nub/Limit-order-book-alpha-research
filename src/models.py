"""Train-only preprocessing, temporal evaluation and calibration diagnostics."""
from __future__ import annotations
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = np.array([-1, 0, 1], dtype=int)


def build_estimator(kind: str, seed: int = 42):
    if kind == "logistic":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=False), StandardScaler(),
                             LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
    if kind == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("Install optional dependency: pip install -e '.[lightgbm]'") from exc
        return make_pipeline(SimpleImputer(strategy="median"), LGBMClassifier(
            n_estimators=180, max_depth=5, num_leaves=15, learning_rate=0.035,
            min_child_samples=80, verbose=-1, random_state=seed, n_jobs=2))
    raise ValueError(f"Unknown model: {kind}")


def probabilities_3(estimator, x):
    """Align columns to [-1,0,+1] even if sklearn model order differs."""
    probs = estimator.predict_proba(x)
    classes = estimator.classes_ if hasattr(estimator, "classes_") else estimator[-1].classes_
    out = np.zeros((len(probs), 3), dtype=float)
    for j, c in enumerate(classes):
        out[:, np.where(CLASSES == c)[0][0]] = probs[:, j]
    return out


def evaluate(y_true, probs: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    pred = CLASSES[np.argmax(probs, axis=1)]
    one_hot = (y_true[:, None] == CLASSES[None, :]).astype(float)
    confidence = probs.max(axis=1)
    correct = (pred == y_true).astype(float)
    # Equal-width confidence bins; empty bins excluded from ECE.
    bin_id = np.minimum((confidence * 10).astype(int), 9)
    ece = sum((bin_id == k).mean() * abs(correct[bin_id == k].mean() - confidence[bin_id == k].mean())
              for k in range(10) if (bin_id == k).any())
    return {
        "n": len(y_true),
        "class_support": {str(int(c)): int((y_true == c).sum()) for c in CLASSES},
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=CLASSES, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, probs, labels=CLASSES)),
        "multiclass_brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "expected_calibration_error": float(ece),
        "confusion_matrix_labels_-1_0_1": confusion_matrix(y_true, pred, labels=CLASSES).tolist(),
    }
