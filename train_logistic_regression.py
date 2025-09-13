"""
A script to train a logistic regression classifier
on top of frozen encoder embeddings.

Author: Mohanad Albughdadi
Created: 2025-09-12
"""

import argparse
from typing import Optional
from pathlib import Path
import json
import time
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from classification_models.logistic_regression import LogisticRegressionModel

# ---------- helpers ----------


def load_npz(
    path: Path,
    x_keys=("x_train", "x_val", "x", "X"),
    y_keys=("y_train", "y_val", "y", "Y"),
):
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    x_key = next((k for k in x_keys if k in data), None)
    y_key = next((k for k in y_keys if k in data), None)
    if x_key is None or y_key is None:
        raise KeyError(f"{path} missing expected keys. Found: {list(data.keys())}")
    return data[x_key], data[y_key], (x_key, y_key)


def coerce_y(y: np.ndarray, multilabel: Optional[str]) -> tuple[np.ndarray, bool]:
    """
    multilabel in {"true","false",None}; None = auto.
    multiclass -> 1D ints; multilabel -> 2D 0/1.
    """
    if multilabel == "true":
        if y.ndim == 1:
            raise ValueError("multilabel=true but y is 1D")
        return y.astype(int), True
    if multilabel == "false":
        if y.ndim == 2:
            rs = y.sum(axis=1)
            if np.all((rs == 1) | (rs == 0)):
                y = y.argmax(axis=1)
            else:
                raise ValueError("multiclass requested but y looks multilabel")
        return y.astype(int).reshape(-1), False

    # auto
    if y.ndim == 1:
        return y.astype(int).reshape(-1), False
    rs = y.sum(axis=1)
    if np.all((rs == 1) | (rs == 0)):  # one-hot or some empties
        return y.argmax(axis=1).astype(int), False
    if set(np.unique(y)).issubset({0, 1}):
        return y.astype(int), True
    raise ValueError("Cannot infer task from y; values not in {0,1} and not one-hot.")


def feature_slice(X: np.ndarray, spec: Optional[str]) -> np.ndarray:
    if not spec:
        return X
    try:
        a, b = spec.split(":")
        a = int(a) if a else None
        b = int(b) if b else None
        return X[:, slice(a, b)]
    except Exception as e:
        raise ValueError(f"Bad --feature-slice '{spec}', use 'start:end'") from e


def now_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser(
        description="Train LogisticRegressionModel on .npz embeddings."
    )
    ap.add_argument("--train", required=True, type=Path, help="Path to train .npz")
    ap.add_argument("--val", type=Path, help="Optional path to val .npz")
    ap.add_argument(
        "--out", type=Path, default=Path("."), help="Output dir (default: '.')"
    )
    ap.add_argument(
        "--name", type=str, default="logreg", help="Base name for artifacts"
    )
    ap.add_argument(
        "--multilabel",
        choices=["true", "false"],
        default=None,
        help="Force task type (default: auto)",
    )
    ap.add_argument(
        "--pool",
        choices=["true", "false"],
        default=None,
        help="Force task type (default: auto)",
    )
    ap.add_argument(
        "--feature-slice", type=str, default=None, help="Column slice 'start:end'"
    )
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    run_id = now_id()

    # Load train
    Xtr, Ytr, train_keys = load_npz(args.train)
    if args.pool:
        Xtr = Xtr.reshape(Xtr.shape[0], 105, 144)
        Xtr = Xtr.mean(axis=1)
    Xtr = feature_slice(Xtr, args.feature_slice)
    Ytr, is_multi = coerce_y(Ytr, args.multilabel)

    print(
        f"[train] X: {Xtr.shape}  Y: {Ytr.shape}  task: {'multilabel' if is_multi else 'multiclass'}"
    )

    # Optional val
    Xval = Yval = None
    if args.val:
        Xval, Yval, val_keys = load_npz(args.val)
        if args.pool:
            Xval = Xval.reshape(Xtr.shape[0], 105, 144)
            Xval = Xval.mean(axis=1)
        Xval = feature_slice(Xval, args.feature_slice)
        Yval, _ = coerce_y(Yval, "true" if is_multi else "false")
        print(f"[val]   X: {Xval.shape}  Y: {Yval.shape}")

    # Train
    model = LogisticRegressionModel(
        multilabel=is_multi, n_jobs=args.n_jobs, verbose=True
    )
    model.fit(Xtr, Ytr)
    print("[info] training complete")

    # Evaluate if val provided
    metrics = {}
    if Xval is not None:
        y_pred = model.predict(Xval)
        if is_multi:
            y_pred = (y_pred > 0.5).astype(int) if y_pred.dtype != int else y_pred
            metrics = {
                "f1_micro": float(
                    f1_score(Yval, y_pred, average="micro", zero_division=0)
                ),
                "f1_macro": float(
                    f1_score(Yval, y_pred, average="macro", zero_division=0)
                ),
                "report": classification_report(
                    Yval, y_pred, output_dict=True, zero_division=0
                ),
            }
            print(
                f"[val] F1-micro: {metrics['f1_micro']:.4f}  F1-macro: {metrics['f1_macro']:.4f}"
            )
        else:
            metrics = {
                "accuracy": float(accuracy_score(Yval, y_pred)),
                "f1_macro": float(f1_score(Yval, y_pred, average="macro")),
                "report": classification_report(Yval, y_pred, output_dict=True),
            }
            print(
                f"[val] Acc: {metrics['accuracy']:.4f}  F1-macro: {metrics['f1_macro']:.4f}"
            )

    # Save artifacts
    model_path = args.out / f"{args.name}_{run_id}.joblib"
    meta_path = args.out / f"{args.name}_{run_id}.json"
    joblib.dump(model, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "train_npz": str(args.train),
                "val_npz": str(args.val) if args.val else None,
                "x_shape_train": tuple(Xtr.shape),
                "y_shape_train": tuple(Ytr.shape),
                "x_shape_val": tuple(Xval.shape) if Xval is not None else None,
                "y_shape_val": tuple(Yval.shape) if Yval is not None else None,
                "multilabel": is_multi,
                "feature_slice": args.feature_slice,
                "n_jobs": args.n_jobs,
                "metrics": metrics,
                "version": "simple-1.0",
                "timestamp": run_id,
            },
            f,
            indent=2,
        )

    print(f"\n[saved] model:    {model_path}")
    print(f"[saved] metadata: {meta_path}")


if __name__ == "__main__":
    main()
