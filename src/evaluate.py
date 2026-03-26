"""
evaluate.py
───────────
Evaluate a saved model against labelled data and produce a full
performance report: classification metrics, confusion matrix, and
ROC curve plot.

Usage
-----
# Evaluate the default model (LR + TF-IDF) on a labelled TSV:
python src/evaluate.py --config config/config.yaml --input data/test_labelled.tsv

# Choose a different model:
python src/evaluate.py --config config/config.yaml \
                       --input data/test_labelled.tsv \
                       --model lr_w2v          # one of: lr_tfidf | lr_w2v | nb_tfidf

# Input has raw 'Abstract' column instead of pre-cleaned text:
python src/evaluate.py --config config/config.yaml \
                       --input data/test_labelled.tsv \
                       --raw

Input file requirements
-----------------------
TSV file with:
  - A text column matching preprocessing.primary_feature_column
    (default: clean_text_Abstract), OR the raw source column if --raw is passed.
  - A 'Label' column with integer values 0 (negative) or 1 (positive).

Output
------
  output/plots/eval_roc_<model>_<timestamp>.png   — ROC curve
  output/predictions/eval_<basename>_<model>_<timestamp>.tsv  — per-sample predictions
  Metrics summary printed to stdout and appended to output/metrics_summary.tsv
"""

import argparse
import logging
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)

from preprocess import full_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

nltk.download("punkt", quiet=True)

SUPPORTED_MODELS = {
    "lr_tfidf": ("lr_tfidf.joblib",  "tfidf_vectorizer.joblib", "tfidf"),
    "nb_tfidf": ("nb_tfidf.joblib",  "tfidf_vectorizer.joblib", "tfidf"),
    "lr_w2v":   ("lr_w2v.joblib",    "word2vec.model",           "w2v"),
}


# ── artefact loading ──────────────────────────────────────────────────────────

def load_artefacts(models_dir: str, model_key: str):
    clf_file, vec_file, vec_type = SUPPORTED_MODELS[model_key]
    clf = joblib.load(os.path.join(models_dir, clf_file))
    logger.info("Loaded classifier : %s", clf_file)

    if vec_type == "tfidf":
        vectorizer = joblib.load(os.path.join(models_dir, vec_file))
        logger.info("Loaded vectoriser  : %s (TF-IDF)", vec_file)
    else:
        from gensim.models import Word2Vec as _W2V
        _W2V.load(os.path.join(models_dir, vec_file))          # validate file exists
        vectorizer = joblib.load(
            os.path.join(models_dir, "mean_embedding_vectorizer.joblib")
        )
        logger.info("Loaded vectoriser  : %s (Word2Vec mean-pooling)", vec_file)

    return clf, vectorizer, vec_type


# ── vectorisation ─────────────────────────────────────────────────────────────

def vectorize(texts: pd.Series, vectorizer, vec_type: str):
    if vec_type == "tfidf":
        return vectorizer.transform(texts)
    tokenized = [nltk.word_tokenize(doc) for doc in texts]
    return vectorizer.transform(tokenized)


# ── reporting helpers ─────────────────────────────────────────────────────────

def print_report(y_true, y_pred, y_prob, model_key: str) -> dict:
    """Print full classification report and return scalar metrics."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Model : {model_key}")
    print(sep)
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}\n")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"  AUC         : {roc_auc:.4f}")
    print(f"  Sensitivity : {sensitivity:.4f}  (Recall for positive class)")
    print(f"  Specificity : {specificity:.4f}  (Recall for negative class)")
    print(f"{sep}\n")

    return {
        "model":       model_key,
        "auc":         round(roc_auc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "fpr":         fpr,
        "tpr":         tpr,
    }


def save_roc_plot(result: dict, plots_dir: str, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        result["fpr"], result["tpr"],
        label=f"{result['model']} (AUC = {result['auc']:.3f})",
        linewidth=2,
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {result['model']}")
    ax.legend(loc="lower right")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, f"eval_roc_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC plot saved → %s", path)


def save_confusion_matrix_plot(y_true, y_pred, model_key: str, plots_dir: str, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["Negative (0)", "Positive (1)"],
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_key}")
    path = os.path.join(plots_dir, f"eval_cm_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix plot saved → %s", path)


def append_metrics(result: dict, metrics_file: str) -> None:
    """Append scalar metrics to the shared metrics summary TSV."""
    row = {k: result[k] for k in ("model", "auc", "sensitivity", "specificity")}
    row["evaluated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_row = pd.DataFrame([row])
    header = not os.path.exists(metrics_file)
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    df_row.to_csv(metrics_file, sep="\t", index=False, mode="a", header=header)
    logger.info("Metrics appended → %s", metrics_file)


# ── main ──────────────────────────────────────────────────────────────────────

def main(cfg: dict, input_path: str, model_key: str, raw: bool) -> None:
    ocfg        = cfg["output"]
    pcfg        = cfg["preprocessing"]
    feature_col = pcfg["primary_feature_column"]
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag         = f"{model_key}_{timestamp}"
    base        = os.path.splitext(os.path.basename(input_path))[0]

    # ── load data ─────────────────────────────────────────────────────────────
    logger.info("Loading labelled data from %s …", input_path)
    df = pd.read_csv(input_path, sep="\t")

    if "Label" not in df.columns:
        raise ValueError(
            "Input file must contain a 'Label' column (0 = negative, 1 = positive)."
        )

    # ── optionally preprocess raw text ────────────────────────────────────────
    if raw:
        raw_col = feature_col.replace("clean_text_", "")
        if raw_col not in df.columns:
            raise ValueError(
                f"--raw specified but column '{raw_col}' not found in input file."
            )
        logger.info("Applying NLP preprocessing to column '%s' …", raw_col)
        df[feature_col] = df[raw_col].apply(full_pipeline)

    if feature_col not in df.columns:
        raise ValueError(
            f"Feature column '{feature_col}' not found. "
            "Either pass --raw or pre-run preprocess.py."
        )

    texts  = df[feature_col].fillna("").astype(str)
    y_true = df["Label"].astype(int)

    logger.info(
        "Label distribution — Positive: %d  |  Negative: %d",
        (y_true == 1).sum(), (y_true == 0).sum(),
    )

    # ── load model and vectorise ──────────────────────────────────────────────
    clf, vectorizer, vec_type = load_artefacts(ocfg["models_dir"], model_key)
    X = vectorize(texts, vectorizer, vec_type)

    # ── predict ───────────────────────────────────────────────────────────────
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]

    # ── report ────────────────────────────────────────────────────────────────
    result = print_report(y_true, y_pred, y_prob, model_key)

    # ── plots ─────────────────────────────────────────────────────────────────
    save_roc_plot(result, ocfg["plots_dir"], tag)
    save_confusion_matrix_plot(y_true, y_pred, model_key, ocfg["plots_dir"], tag)

    # ── persist predictions ───────────────────────────────────────────────────
    df["predicted_label"] = y_pred
    df["predicted_prob"]  = y_prob
    df["correct"]         = (y_pred == y_true.values).astype(int)

    os.makedirs(ocfg["predictions_dir"], exist_ok=True)
    out_path = os.path.join(
        ocfg["predictions_dir"], f"eval_{base}_{tag}.tsv"
    )
    df.to_csv(out_path, sep="\t", index=False)
    logger.info("Per-sample predictions saved → %s", out_path)

    # ── append to shared metrics summary ─────────────────────────────────────
    append_metrics(result, ocfg["metrics_file"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on labelled data."
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to labelled TSV file (must contain a 'Label' column)"
    )
    parser.add_argument(
        "--model", default="lr_tfidf",
        choices=list(SUPPORTED_MODELS.keys()),
        help="Which saved model to evaluate (default: lr_tfidf)"
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Input contains raw text; apply NLP pipeline before evaluation"
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    main(cfg, args.input, args.model, args.raw)
