"""Module I — Train on Synthetic, Test on Real (TSTR).

Trains a scikit-learn RandomForestClassifier on synthetic data and
evaluates it on the original (real) held-out data.  Provides an
empirical measure of whether the synthetic data preserves the
predictive signal required for downstream ML tasks.

Target column priority
----------------------
1. is_deceased   (patients)
2. is_chronic    (conditions)
3. is_active     (medications)
4. is_readmission, is_emergency
5. Any bool column not in EXCLUDED_COLUMNS
6. Any 0/1 integer column not in EXCLUDED_COLUMNS

If no suitable target is found, the table is skipped gracefully.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

from .config import EXCLUDED_COLUMNS
from .loader import TablePair

logger = logging.getLogger(__name__)

_TARGET_PRIORITY: tuple[str, ...] = (
    "is_deceased", "is_chronic", "is_active",
    "is_readmission", "is_emergency",
)

_MIN_SAMPLES = 50
_RF_PARAMS   = dict(n_estimators=100, random_state=42, n_jobs=-1, max_depth=8)


def run_tstr_evaluation(
    pairs:      dict[str, TablePair],
    output_dir: Path,
) -> dict[str, dict]:
    """Run TSTR for every table where a binary target exists.

    Parameters
    ----------
    pairs      : table-name → (original_df, synthetic_df)
    output_dir : outputs/evaluation/ — where CSV/JSON are written

    Returns
    -------
    dict keyed by table name::

        {
            "status":         "ok" | "no_target" | "missing_data" |
                              "insufficient_data" | "error",
            "target_column":  str,
            "n_train":        int,
            "n_test":         int,
            "accuracy":       float,
            "precision":      float,
            "recall":         float,
            "f1":             float,
            "roc_auc":        float | None,
            "class_balance_train": {0: int, 1: int},
            "class_balance_test":  {0: int, 1: int},
        }
    """
    all_results: dict[str, dict] = {}

    for table_name, (original, synthetic) in pairs.items():
        if original is None or synthetic is None:
            logger.warning("[TSTR] %s — missing data, skipped", table_name)
            all_results[table_name] = {"status": "missing_data"}
            continue

        result = _evaluate_table(original, synthetic, table_name)
        _log_result(table_name, result)
        all_results[table_name] = result

    _save_csv(output_dir, all_results)
    _save_json(output_dir, all_results)
    return all_results


# ── Per-table logic ───────────────────────────────────────────────────────────

def _evaluate_table(
    original:   pd.DataFrame,
    synthetic:  pd.DataFrame,
    table_name: str,
) -> dict:
    target = _find_target(original, synthetic)
    if target is None:
        return {"status": "no_target"}

    if len(synthetic) < _MIN_SAMPLES or len(original) < _MIN_SAMPLES:
        return {"status": "insufficient_data", "target_column": target}

    try:
        X_train, y_train = _prepare(synthetic, target)
        X_test,  y_test  = _prepare(original,  target,
                                    fit_cols=X_train.columns.tolist())

        if X_train.empty or X_test.empty or len(y_train) < _MIN_SAMPLES:
            return {"status": "insufficient_data", "target_column": target}

        # Align columns — test may be missing dummies that exist in train
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        if y_train.nunique() < 2:
            return {
                "status":        "insufficient_data",
                "target_column": target,
                "note":          "single class in synthetic training set",
            }

        clf = RandomForestClassifier(**_RF_PARAMS)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        result: dict = {
            "status":        "ok",
            "target_column": target,
            "n_train":       int(len(X_train)),
            "n_test":        int(len(X_test)),
            "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(
                             y_test, y_pred, average="binary", zero_division=0)), 4),
            "recall":    round(float(recall_score(
                             y_test, y_pred, average="binary", zero_division=0)), 4),
            "f1":        round(float(f1_score(
                             y_test, y_pred, average="binary", zero_division=0)), 4),
            "class_balance_train": {
                str(k): int(v) for k, v in y_train.value_counts().to_dict().items()
            },
            "class_balance_test": {
                str(k): int(v) for k, v in y_test.value_counts().to_dict().items()
            },
        }

        if y_test.nunique() >= 2:
            try:
                y_prob = clf.predict_proba(X_test)[:, 1]
                result["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
            except Exception:
                result["roc_auc"] = None
        else:
            result["roc_auc"] = None

        return result

    except Exception as exc:
        logger.warning("[TSTR] %s — error: %s", table_name, exc)
        return {"status": "error", "error": str(exc), "target_column": target}


# ── Target detection ──────────────────────────────────────────────────────────

def _find_target(
    original:  pd.DataFrame,
    synthetic: pd.DataFrame,
) -> str | None:
    """Find the best binary classification target in both frames."""
    both = set(original.columns) & set(synthetic.columns)

    # 1. Preferred names
    for name in _TARGET_PRIORITY:
        if name in both:
            return name

    # 2. Boolean columns
    for col in sorted(both):
        if col in EXCLUDED_COLUMNS:
            continue
        if pd.api.types.is_bool_dtype(original[col]):
            return col

    # 3. Integer columns with values exactly in {0, 1}
    for col in sorted(both):
        if col in EXCLUDED_COLUMNS:
            continue
        if pd.api.types.is_integer_dtype(original[col]):
            vals = set(original[col].dropna().unique())
            if vals <= {0, 1} and len(vals) == 2:
                return col

    return None


# ── Feature preparation ───────────────────────────────────────────────────────

def _prepare(
    df:       pd.DataFrame,
    target:   str,
    fit_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) pair from a DataFrame."""
    drop = (set(EXCLUDED_COLUMNS) | {target}) & set(df.columns)
    X    = df.drop(columns=list(drop)).copy()

    # Bool → int
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)

    # Encode objects
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str).fillna("__NULL__"))

    # Fill remaining NaN
    X = X.fillna(0)

    # Restrict to training columns if aligning test set
    if fit_cols is not None:
        available = [c for c in fit_cols if c in X.columns]
        X = X[available] if available else X.iloc[:, :0]  # empty frame

    # Build target vector
    y = df[target].copy()
    if pd.api.types.is_bool_dtype(y):
        y = y.astype(int)
    else:
        y = pd.to_numeric(y, errors="coerce").astype("Int64")

    valid = y.notna()
    return X.loc[valid].reset_index(drop=True), y[valid].astype(int).reset_index(drop=True)


# ── File I/O ──────────────────────────────────────────────────────────────────

def _save_csv(output_dir: Path, results: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for table, r in results.items():
        rows.append({
            "table":         table,
            "status":        r.get("status"),
            "target_column": r.get("target_column"),
            "n_train":       r.get("n_train"),
            "n_test":        r.get("n_test"),
            "accuracy":      r.get("accuracy"),
            "precision":     r.get("precision"),
            "recall":        r.get("recall"),
            "f1":            r.get("f1"),
            "roc_auc":       r.get("roc_auc"),
        })
    path = output_dir / "tstr_results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Saved: %s", path)


def _save_json(output_dir: Path, results: dict) -> None:
    path = output_dir / "tstr_results.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Saved: %s", path)


def _log_result(table: str, result: dict) -> None:
    status = result.get("status")
    if status != "ok":
        logger.info("[TSTR] %-14s  %s", table, status)
        return
    logger.info(
        "[TSTR] %-14s  target=%-15s  acc=%.3f  f1=%.3f  auc=%s",
        table,
        result.get("target_column", "?"),
        result.get("accuracy", 0),
        result.get("f1", 0),
        f"{result['roc_auc']:.3f}" if result.get("roc_auc") is not None else "N/A",
    )
