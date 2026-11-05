# backend/app/ml/predictors/train_logreg.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict, Any
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold


LABELS = [0, 1, 2]  # 0=HomeWin, 1=Draw, 2=AwayWin
CLASS_NAMES = {0: "HomeWin", 1: "Draw", 2: "AwayWin"}


def _load_split(splits_dir: Path, name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    path = splits_dir / f"{name}_normalized.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError(f"{path} must contain a 'target' column.")

    y = df["target"].astype(int).to_numpy()
    Xdf = df.drop(columns=["target"])

    feature_cols = list(Xdf.columns)
    X = Xdf.to_numpy(dtype=float)
    return X, y, feature_cols


def _print_class_balance(y: np.ndarray, title: str) -> None:
    vals, cnts = np.unique(y, return_counts=True)
    total = cnts.sum()
    print(f"\n{title} class distribution:")
    for v, c in zip(vals, cnts):
        print(f"  {int(v)} ({CLASS_NAMES.get(int(v), str(v))}): {int(c)}  ({c/total:.2%})")


def _evaluate(name: str, model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    proba = model.predict_proba(X)
    yhat = proba.argmax(axis=1)

    acc = accuracy_score(y, yhat)
    ll = log_loss(y, proba, labels=LABELS)

    cm = confusion_matrix(y, yhat, labels=LABELS).tolist()
    report = classification_report(
        y, yhat, labels=LABELS, digits=4, zero_division=0, output_dict=True
    )

    print("\n" + "=" * 70)
    print(f"[{name}] accuracy={acc:.4f}  logloss={ll:.4f}\n")
    print("Classification report:")
    print(classification_report(y, yhat, labels=LABELS, digits=4, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y, yhat, labels=LABELS))

    return {
        "accuracy": float(acc),
        "logloss": float(ll),
        "confusion_matrix": cm,
        "report": report,
    }


def _top_features_per_class(model: LogisticRegression, feature_cols: List[str], k: int = 8) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not hasattr(model, "coef_"):
        return out

    W = model.coef_
    if W.ndim != 2 or W.shape[1] != len(feature_cols):
        return out

    for ci, class_id in enumerate(model.classes_):
        weights = W[ci]
        idx = np.argsort(weights)[-k:][::-1]
        feats = [{"feature": feature_cols[j], "weight": float(weights[j])} for j in idx]
        out.append(
            {
                "class": int(class_id),
                "class_name": CLASS_NAMES.get(int(class_id), str(class_id)),
                "top_positive": feats,
            }
        )
    return out


def _load_metadata(path: Path) -> Dict[str, Any]:
    # Backward compatible with your current format:
    # { "default": "...", "perLeague": {} }
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"default": "logreg_v1", "perLeague": {}}
    return {"default": "logreg_v1", "perLeague": {}}


def _save_metadata(path: Path, meta: Dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    app_dir = Path(__file__).resolve().parents[2]  # .../backend/app
    splits_dir = app_dir / "data" / "splits"
    artifacts_dir = app_dir / "ml" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load splits (keep test untouched)
    X_train, y_train, feature_cols_train = _load_split(splits_dir, "train")
    X_valid, y_valid, feature_cols_valid = _load_split(splits_dir, "valid")
    X_test, y_test, feature_cols_test = _load_split(splits_dir, "test")

    if feature_cols_train != feature_cols_valid or feature_cols_train != feature_cols_test:
        raise ValueError("Feature columns differ across splits. Check your dataset build output.")

    _print_class_balance(y_train, "TRAIN")
    _print_class_balance(y_valid, "VALID")
    _print_class_balance(y_test, "TEST")

    # Combine train+valid for CV selection ONLY (test remains final holdout)
    X_cv = np.vstack([X_train, X_valid])
    y_cv = np.concatenate([y_train, y_valid])

    # ---- Cross-validation config ----
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    C_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

    model_id = "logreg_v2"  # bump this when you retrain again

    # Total timing: CV + final train + evaluations
    t0_total = time.perf_counter()

    print("\n" + "=" * 70)
    print(f"Running {n_splits}-Fold Stratified CV on (train+valid)")
    print(f"C grid: {C_grid}\n")

    cv_rows: List[Dict[str, Any]] = []
    best_C = None
    best_score = -1.0

    t0_cv = time.perf_counter()
    for C in C_grid:
        fold_scores = []
        fold_draw_f1 = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_cv, y_cv), 1):
            X_tr, X_va = X_cv[tr_idx], X_cv[va_idx]
            y_tr, y_va = y_cv[tr_idx], y_cv[va_idx]

            model = LogisticRegression(
                solver="lbfgs",
                max_iter=4000,
                C=C,
                class_weight="balanced",
                random_state=42,
            )
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_va)
            macro_f1 = f1_score(y_va, y_pred, average="macro", zero_division=0)
            draw_f1 = f1_score((y_va == 1).astype(int), (y_pred == 1).astype(int), zero_division=0)

            fold_scores.append(float(macro_f1))
            fold_draw_f1.append(float(draw_f1))

        mean_macro = float(np.mean(fold_scores))
        std_macro = float(np.std(fold_scores))
        mean_draw = float(np.mean(fold_draw_f1))

        cv_rows.append(
            {
                "C": float(C),
                "mean_macro_f1": mean_macro,
                "std_macro_f1": std_macro,
                "mean_draw_f1": mean_draw,
            }
        )

        print(f"C={C:<5}  macro-F1={mean_macro:.4f} ± {std_macro:.4f}   draw-F1={mean_draw:.4f}")

        # Select by macro-F1 (better balance across classes incl. draw)
        if mean_macro > best_score:
            best_score = mean_macro
            best_C = float(C)

    cv_seconds = time.perf_counter() - t0_cv

    if best_C is None:
        raise RuntimeError("CV failed: no best C selected.")

    print("\n" + "=" * 70)
    print(f"Selected best C={best_C} by CV macro-F1={best_score:.4f}")
    print(f"CV time: {cv_seconds:.2f} seconds")

    # ---- Final train on ALL train+valid with best_C ----
    t0_fit = time.perf_counter()
    final_model = LogisticRegression(
        solver="lbfgs",
        max_iter=4000,
        C=best_C,
        class_weight="balanced",
        random_state=42,
    )
    final_model.fit(X_cv, y_cv)
    fit_seconds = time.perf_counter() - t0_fit

    print("\nFinal fit time (train+valid): {:.2f} seconds".format(fit_seconds))

    # ---- Evaluate final model ----
    # NOTE: train/valid here are the original splits; useful to compare
    m_train = _evaluate("train (final)", final_model, X_train, y_train)
    m_valid = _evaluate("valid (final)", final_model, X_valid, y_valid)
    m_test = _evaluate("test (final)", final_model, X_test, y_test)

    top_feats = _top_features_per_class(final_model, feature_cols_train, k=8)

    # ---- Save model ----
    out_path = artifacts_dir / f"{model_id}.joblib"
    joblib.dump(final_model, out_path)
    print("\nSaved model:", out_path)

    # Save feature order (needed for inference)
    feature_order_path = app_dir / "ml" / "gamedb" / "feature_order.json"
    feature_order_path.parent.mkdir(parents=True, exist_ok=True)
    feature_order_path.write_text(
        pd.Series(feature_cols_train).to_json(orient="values"),
        encoding="utf-8",
    )
    print("Saved feature order:", feature_order_path)

    # ---- Update metadata.json (backward compatible) ----
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    total_seconds = time.perf_counter() - t0_total

    metadata_path = artifacts_dir / "metadata.json"
    meta = _load_metadata(metadata_path)

    meta.setdefault("perLeague", {})
    meta.setdefault("models", {})

    meta["models"][model_id] = {
        "type": "logistic_regression",
        "created_at": created_at,
        "timing_seconds": {
            "cv": round(float(cv_seconds), 4),
            "final_fit": round(float(fit_seconds), 4),
            "total": round(float(total_seconds), 4),
        },
        "data": {
            "n_train": int(len(y_train)),
            "n_valid": int(len(y_valid)),
            "n_test": int(len(y_test)),
            "n_train_plus_valid": int(len(y_cv)),
            "n_features": int(X_train.shape[1]),
            "labels": {"0": "HomeWin", "1": "Draw", "2": "AwayWin"},
        },
        "selection": {
            "criterion": "cv_macro_f1",
            "best_C": float(best_C),
            "cv_folds": int(n_splits),
            "cv_grid": [float(x) for x in C_grid],
            "cv_results": cv_rows,
        },
        "hyperparams": {
            "solver": "lbfgs",
            "class_weight": "balanced",
            "C": float(best_C),
            "max_iter": 4000,
            "random_state": 42,
        },
        "metrics": {
            "train": m_train,
            "valid": m_valid,
            "test": m_test,
        },
        "explainability": {
            "top_features_per_class": top_feats,
        },
        "artifacts": {
            "model_joblib": str(out_path.as_posix()),
            "feature_order_json": str(feature_order_path.as_posix()),
        },
    }

    # Set as default so your app can pick it
    meta["default"] = model_id

    _save_metadata(metadata_path, meta)
    print("Updated metadata registry:", metadata_path)

    print("\n" + "=" * 70)
    print(f"TOTAL training pipeline time: {total_seconds:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
