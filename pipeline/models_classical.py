"""Classical ML models: Logistic Regression, SVM, Random Forest, XGBoost, MLP.
Subject-dependent (stratified k-fold within subject) and subject-independent
(Leave-One-Subject-Out) evaluation protocols.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from pipeline import config, features_matrix

MODEL_FACTORIES = {
    "LogisticRegression": lambda seed: LogisticRegression(max_iter=2000, random_state=seed),
    "SVM": lambda seed: SVC(kernel="rbf", probability=True, random_state=seed),
    "RandomForest": lambda seed: RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
    "XGBoost": lambda seed: XGBClassifier(
        eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0,
    ),
    "MLP": lambda seed: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed),
}


def _fit_eval_fold(model_factory, X_train, y_train, X_test, y_test, seed):
    scaler = features_matrix.standardize_fit(X_train)
    Xtr = features_matrix.standardize_apply(scaler, X_train)
    Xte = features_matrix.standardize_apply(scaler, X_test)

    model = model_factory(seed)
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(Xte)[:, 1]
    else:
        y_proba = model.decision_function(Xte)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan,
    }
    return metrics, y_pred, y_proba


def run_cv(X, y, groups, splits, protocol_name, seed=config.SEED, model_names=None):
    """`splits` is an iterable of (train_idx, test_idx, fold_label) triples.
    Returns (results_df, predictions) where predictions[model_name] =
    {"y_true": ..., "y_pred": ..., "y_proba": ...} concatenated across folds.
    """
    model_names = model_names or list(MODEL_FACTORIES.keys())
    rows = []
    predictions = {m: {"y_true": [], "y_pred": [], "y_proba": []} for m in model_names}

    for train_idx, test_idx, fold_label in splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(X_test) == 0:
            continue

        for name in model_names:
            metrics, y_pred, y_proba = _fit_eval_fold(
                MODEL_FACTORIES[name], X_train, y_train, X_test, y_test, seed,
            )
            rows.append({"protocol": protocol_name, "model": name, "fold": fold_label, **metrics})
            predictions[name]["y_true"].append(y_test)
            predictions[name]["y_pred"].append(y_pred)
            predictions[name]["y_proba"].append(y_proba)

    for name in model_names:
        for k in ("y_true", "y_pred", "y_proba"):
            predictions[name][k] = np.concatenate(predictions[name][k]) if predictions[name][k] else np.array([])

    return pd.DataFrame(rows), predictions


def subject_dependent_splits(y, subjects, n_splits=5, seed=config.SEED):
    """Stratified k-fold within each subject; fold_label = f"{subject}_fold{k}"."""
    for subj in sorted(set(subjects)):
        idx = np.where(subjects == subj)[0]
        y_sub = y[idx]
        if len(np.unique(y_sub)) < 2 or len(idx) < n_splits:
            continue
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(skf.split(idx, y_sub)):
            yield idx[tr], idx[te], f"{subj}_fold{k}"


def subject_independent_splits(y, subjects):
    """Leave-one-subject-out; fold_label = held-out subject name."""
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(np.zeros(len(y)), y, groups=subjects):
        yield tr, te, str(subjects[te[0]])


def plot_model_comparison(results_df, out_path):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    summary = results_df.groupby(["protocol", "model"])[metrics].mean().reset_index()
    protocols = summary["protocol"].unique()

    fig, axes = plt.subplots(1, len(protocols), figsize=(7 * len(protocols), 5), squeeze=False)
    for i, proto in enumerate(protocols):
        ax = axes[0][i]
        sub = summary[summary["protocol"] == proto].set_index("model")[metrics]
        sub.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title(proto)
        ax.set_ylim(0, 1)
        ax.set_ylabel("score")
        ax.legend(fontsize=7, loc="lower right")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Model comparison (mean across folds)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[models_classical] saved {out_path}")


def plot_confusion_matrices(predictions, protocol_name, out_path):
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, preds) in zip(axes, predictions.items()):
        if len(preds["y_true"]) == 0:
            ax.set_visible(False)
            continue
        cm = confusion_matrix(preds["y_true"], preds["y_pred"])
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["non_reading", "reading"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["non_reading", "reading"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(name, fontsize=9)
    fig.suptitle(f"Confusion matrices — {protocol_name}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[models_classical] saved {out_path}")


def plot_roc_curves(predictions, protocol_name, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, preds in predictions.items():
        if len(preds["y_true"]) == 0 or len(np.unique(preds["y_true"])) < 2:
            continue
        fpr, tpr, _ = roc_curve(preds["y_true"], preds["y_proba"])
        auc = roc_auc_score(preds["y_true"], preds["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})", lw=1.5)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curves — {protocol_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[models_classical] saved {out_path}")
