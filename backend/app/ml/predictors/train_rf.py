# backend/app/ml/predictors/train_rf.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict, Any
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

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


def _evaluate(name: str, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    proba = model.predict_proba(X)
    yhat = proba.argmax(axis=1)

    acc = accuracy_score(y, yhat)
    ll = log_loss(y, proba, labels=LABELS)
    
    # Calculate precision, recall, F1 for each class and macro averages
    precision_per_class = precision_score(y, yhat, labels=LABELS, average=None, zero_division=0)
    recall_per_class = recall_score(y, yhat, labels=LABELS, average=None, zero_division=0)
    f1_per_class = f1_score(y, yhat, labels=LABELS, average=None, zero_division=0)
    
    # Macro averages (balanced across classes)
    precision_macro = precision_score(y, yhat, labels=LABELS, average='macro', zero_division=0)
    recall_macro = recall_score(y, yhat, labels=LABELS, average='macro', zero_division=0)
    f1_macro = f1_score(y, yhat, labels=LABELS, average='macro', zero_division=0)
    
    # Weighted averages (weighted by support)
    precision_weighted = precision_score(y, yhat, labels=LABELS, average='weighted', zero_division=0)
    recall_weighted = recall_score(y, yhat, labels=LABELS, average='weighted', zero_division=0)
    f1_weighted = f1_score(y, yhat, labels=LABELS, average='weighted', zero_division=0)

    cm = confusion_matrix(y, yhat, labels=LABELS).tolist()
    report_dict = classification_report(
        y, yhat, labels=LABELS, digits=4, zero_division=0, output_dict=True
    )

    print("\n" + "=" * 70)
    print(f"[{name}] accuracy={acc:.4f}  logloss={ll:.4f}")
    print(f"Precision (macro)={precision_macro:.4f}  Recall (macro)={recall_macro:.4f}  F1 (macro)={f1_macro:.4f}")
    print(f"Precision (weighted)={precision_weighted:.4f}  Recall (weighted)={recall_weighted:.4f}  F1 (weighted)={f1_weighted:.4f}\n")
    
    print("Per-class metrics:")
    for idx, label in enumerate(LABELS):
        print(f"  Class {label} ({CLASS_NAMES[label]}): "
              f"Precision={precision_per_class[idx]:.4f}, "
              f"Recall={recall_per_class[idx]:.4f}, "
              f"F1={f1_per_class[idx]:.4f}")
    
    print("\nClassification report:")
    print(classification_report(y, yhat, labels=LABELS, digits=4, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y, yhat, labels=LABELS))

    return {
        "accuracy": float(acc),
        "logloss": float(ll),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "precision_per_class": {CLASS_NAMES[label]: float(precision_per_class[i]) for i, label in enumerate(LABELS)},
        "recall_per_class": {CLASS_NAMES[label]: float(recall_per_class[i]) for i, label in enumerate(LABELS)},
        "f1_per_class": {CLASS_NAMES[label]: float(f1_per_class[i]) for i, label in enumerate(LABELS)},
        "confusion_matrix": cm,
        "report": report_dict,
    }


def _load_metadata(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"default": "rf_v1", "perLeague": {}}


def _save_metadata(path: Path, meta: Dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> Dict[str, Any]:
    app_dir = Path(__file__).resolve().parents[2]  # .../backend/app
    splits_dir = app_dir / "data" / "splits"
    artifacts_dir = app_dir / "ml" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Start total timer
    t0_total = time.perf_counter()

    # Load splits
    X_train, y_train, feat_train = _load_split(splits_dir, "train")
    X_valid, y_valid, feat_valid = _load_split(splits_dir, "valid")
    X_test, y_test, feat_test = _load_split(splits_dir, "test")

    if feat_train != feat_valid or feat_train != feat_test:
        raise ValueError("Feature columns differ across splits. Check dataset build output.")

    _print_class_balance(y_train, "TRAIN")
    _print_class_balance(y_valid, "VALID")
    _print_class_balance(y_test, "TEST")

    # Combine train+valid for CV selection (keep test untouched)
    X_cv = np.vstack([X_train, X_valid])
    y_cv = np.concatenate([y_train, y_valid])

    # -------------------------
    # Cross-validation settings
    # -------------------------
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Small, practical grid (fast but meaningful)
    grid = [
        {"min_samples_leaf": 2, "min_samples_split": 6},
        {"min_samples_leaf": 2, "min_samples_split": 10},
        {"min_samples_leaf": 5, "min_samples_split": 6},
        {"min_samples_leaf": 5, "min_samples_split": 10},
    ]

    # Fixed RF settings (you can expand later)
    n_estimators = 600
    max_features = "sqrt"
    class_weight = "balanced_subsample"
    random_state = 42
    n_jobs = -1

    model_id = "rf_v2"  # bump version

    print("\n" + "=" * 70)
    print(f"Running {n_splits}-Fold Stratified CV on (train+valid)")
    print(f"Grid size: {len(grid)} (tuning min_samples_leaf / min_samples_split)")
    print("Selection metric: macro-F1 (balanced across Home/Draw/Away)\n")

    results: List[Dict[str, Any]] = []

    t0_cv = time.perf_counter()
    
    for params in grid:
        leaf = params["min_samples_leaf"]
        split = params["min_samples_split"]

        fold_macro = []
        fold_draw = []
        fold_time = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_cv, y_cv), 1):
            X_tr, X_va = X_cv[tr_idx], X_cv[va_idx]
            y_tr, y_va = y_cv[tr_idx], y_cv[va_idx]

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=leaf,
                min_samples_split=split,
                max_features=max_features,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=n_jobs,
            )

            t_fit = time.perf_counter()
            model.fit(X_tr, y_tr)
            fold_time.append(time.perf_counter() - t_fit)

            y_pred = model.predict(X_va)

            macro_f1 = f1_score(y_va, y_pred, average="macro", zero_division=0)
            # draw-F1 as a diagnostic
            draw_f1 = f1_score((y_va == 1).astype(int), (y_pred == 1).astype(int), zero_division=0)

            fold_macro.append(float(macro_f1))
            fold_draw.append(float(draw_f1))

        mean_macro = float(np.mean(fold_macro))
        std_macro = float(np.std(fold_macro))
        mean_draw = float(np.mean(fold_draw))
        mean_fit_s = float(np.mean(fold_time))

        row = {
            "params": {"min_samples_leaf": leaf, "min_samples_split": split},
            "mean_macro_f1": mean_macro,
            "std_macro_f1": std_macro,
            "mean_draw_f1": mean_draw,
            "mean_fit_seconds_per_fold": mean_fit_s,
        }
        results.append(row)

        print(
            f"leaf={leaf:<2} split={split:<2}  "
            f"macro-F1={mean_macro:.4f} ± {std_macro:.4f}   "
            f"draw-F1={mean_draw:.4f}   (avg fit {mean_fit_s:.2f}s/fold)"
        )

    cv_seconds = time.perf_counter() - t0_cv

    # Pick best by macro-F1, tie-break by draw-F1
    results_sorted = sorted(results, key=lambda r: (r["mean_macro_f1"], r["mean_draw_f1"]), reverse=True)
    best = results_sorted[0]
    best_leaf = best["params"]["min_samples_leaf"]
    best_split = best["params"]["min_samples_split"]

    print("\n" + "=" * 70)
    print(f"Selected best params: min_samples_leaf={best_leaf}, min_samples_split={best_split}")
    print(f"CV time: {cv_seconds:.2f} seconds")

    # -------------------------
    # Final training on train+valid
    # -------------------------
    t0_fit_final = time.perf_counter()
    final_model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=best_leaf,
        min_samples_split=best_split,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    final_model.fit(X_cv, y_cv)
    final_fit_seconds = time.perf_counter() - t0_fit_final

    print(f"\nFinal fit time (train+valid): {final_fit_seconds:.2f} seconds")

    # Evaluate on TEST only (final estimate)
    m_test = _evaluate("TEST (final rf)", final_model, X_test, y_test)

    total_seconds = time.perf_counter() - t0_total
    
    # Calculate validation metrics on validation set (not used during training)
    print("\n" + "=" * 70)
    print("VALIDATION SET METRICS:")
    m_valid = _evaluate("VALIDATION", final_model, X_valid, y_valid)
    
    print("\n" + "=" * 70)
    print(f"TOTAL pipeline time (CV + final fit + test eval): {total_seconds:.2f} seconds")

    # Save model
    out_path = artifacts_dir / f"{model_id}.joblib"
    joblib.dump(final_model, out_path)
    print("\nSaved model:", out_path)

    # Save feature order
    feature_order_path = app_dir / "ml" / "gamedb" / "feature_order.json"
    feature_order_path.parent.mkdir(parents=True, exist_ok=True)
    feature_order_path.write_text(
        pd.Series(feat_train).to_json(orient="values"),
        encoding="utf-8",
    )
    print("Saved feature order:", feature_order_path)

    # Update metadata registry
    metadata_path = artifacts_dir / "metadata.json"
    meta = _load_metadata(metadata_path)
    meta.setdefault("perLeague", {})
    meta.setdefault("models", {})

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    meta["models"][model_id] = {
        "type": "random_forest",
        "created_at": created_at,
        "timing_seconds": {
            "cv": round(float(cv_seconds), 4),
            "final_fit": round(float(final_fit_seconds), 4),
            "total": round(float(total_seconds), 4),
        },
        "hyperparams": {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "class_weight": class_weight,
            "random_state": random_state,
            "min_samples_leaf": int(best_leaf),
            "min_samples_split": int(best_split),
            "cv_folds": int(n_splits),
            "selection_metric": "macro_f1",
            "grid": grid,
        },
        "cv_results": results,
        "metrics": {
            "validation": m_valid,
            "test": m_test
        },
        "artifacts": {
            "model_joblib": str(out_path.as_posix()),
            "feature_order_json": str(feature_order_path.as_posix()),
        },
    }

    # ✅ Choose model only by name
    meta["default"] = model_id
    _save_metadata(metadata_path, meta)
    print("Updated metadata registry:", metadata_path)

    # Return final metrics summary
    final_summary = {
        "validation": {
            "accuracy_percent": m_valid["accuracy"] * 100,
            "precision_macro_percent": m_valid["precision_macro"] * 100,
            "recall_macro_percent": m_valid["recall_macro"] * 100,
            "f1_macro_percent": m_valid["f1_macro"] * 100,
            "precision_weighted_percent": m_valid["precision_weighted"] * 100,
            "recall_weighted_percent": m_valid["recall_weighted"] * 100,
            "f1_weighted_percent": m_valid["f1_weighted"] * 100,
        },
        "test": {
            "accuracy_percent": m_test["accuracy"] * 100,
            "precision_macro_percent": m_test["precision_macro"] * 100,
            "recall_macro_percent": m_test["recall_macro"] * 100,
            "f1_macro_percent": m_test["f1_macro"] * 100,
            "precision_weighted_percent": m_test["precision_weighted"] * 100,
            "recall_weighted_percent": m_test["recall_weighted"] * 100,
            "f1_weighted_percent": m_test["f1_weighted"] * 100,
        },
        "timing": {
            "cv_seconds": cv_seconds,
            "final_fit_seconds": final_fit_seconds,
            "total_seconds": total_seconds,
        },
        "best_params": {
            "min_samples_leaf": best_leaf,
            "min_samples_split": best_split,
        },
        "model_id": model_id,
        "created_at": created_at,
    }
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL TRAINING SUMMARY")
    print("=" * 70)
    print(f"Model ID: {model_id}")
    print(f"Created at: {created_at}")
    print(f"Best parameters: min_samples_leaf={best_leaf}, min_samples_split={best_split}")
    print(f"Total training time: {total_seconds:.2f} seconds")
    print("\nVALIDATION SET PERFORMANCE (%):")
    print(f"  Accuracy: {m_valid['accuracy']*100:.2f}%")
    print(f"  Precision (macro): {m_valid['precision_macro']*100:.2f}%")
    print(f"  Recall (macro): {m_valid['recall_macro']*100:.2f}%")
    print(f"  F1 Score (macro): {m_valid['f1_macro']*100:.2f}%")
    print(f"  Precision (weighted): {m_valid['precision_weighted']*100:.2f}%")
    print(f"  Recall (weighted): {m_valid['recall_weighted']*100:.2f}%")
    print(f"  F1 Score (weighted): {m_valid['f1_weighted']*100:.2f}%")
    print("\nTEST SET PERFORMANCE (%):")
    print(f"  Accuracy: {m_test['accuracy']*100:.2f}%")
    print(f"  Precision (macro): {m_test['precision_macro']*100:.2f}%")
    print(f"  Recall (macro): {m_test['recall_macro']*100:.2f}%")
    print(f"  F1 Score (macro): {m_test['f1_macro']*100:.2f}%")
    print(f"  Precision (weighted): {m_test['precision_weighted']*100:.2f}%")
    print(f"  Recall (weighted): {m_test['recall_weighted']*100:.2f}%")
    print(f"  F1 Score (weighted): {m_test['f1_weighted']*100:.2f}%")
    print("=" * 70)
    
    return final_summary


if __name__ == "__main__":
    results = main()
    # You can access the results programmatically:
    # print(f"\nValidation Accuracy: {results['validation']['accuracy_percent']:.2f}%")
    # print(f"Test F1 Score: {results['test']['f1_macro_percent']:.2f}%")
    # print(f"Total Time: {results['timing']['total_seconds']:.2f} seconds")