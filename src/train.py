"""
train.py
────────
Train three text classification models:
  1. Logistic Regression  + TF-IDF
  2. Logistic Regression  + Word2Vec (mean pooling)
  3. Naïve Bayes (Multinomial) + TF-IDF

Saves: trained model objects, evaluation metrics TSV, ROC-curve plots.

Usage
-----
python src/train.py --config config/config.yaml
"""

import argparse
import logging
import os
import pickle

import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import yaml
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

nltk.download("punkt", quiet=True)


# ── Word2Vec mean-pooling vectoriser ─────────────────────────────────────────

class MeanEmbeddingVectorizer:
    """
    Average all word vectors in a document to produce a fixed-length
    document embedding using a pre-trained Word2Vec model.
    """

    def __init__(self, word2vec: dict):
        self.word2vec = word2vec
        self.dim = len(next(iter(word2vec.values())))  # infer dim from any entry

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([
            np.mean(
                [self.word2vec[w] for w in words if w in self.word2vec]
                or [np.zeros(self.dim)],
                axis=0,
            )
            for words in X
        ])


# ── helpers ───────────────────────────────────────────────────────────────────

def _evaluate(name: str, y_true, y_pred, y_prob) -> dict:
    """Print and return classification metrics."""
    logger.info("── %s ──────────────────────────────────", name)
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.4f}\n")
    return {"model": name, "auc": roc_auc, "fpr": fpr, "tpr": tpr}


def _plot_roc(results: list, plots_dir: str) -> None:
    """Save a combined ROC-curve figure for all models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in results:
        ax.plot(r["fpr"], r["tpr"], label=f"{r['model']} (AUC = {r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – all models")
    ax.legend(loc="lower right")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, "roc_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC plot saved → %s", path)


def _save_metrics(results: list, metrics_file: str) -> None:
    rows = [{"model": r["model"], "auc": round(r["auc"], 4)} for r in results]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    df.to_csv(metrics_file, sep="\t", index=False)
    logger.info("Metrics saved → %s", metrics_file)


# ── main ──────────────────────────────────────────────────────────────────────

def main(cfg: dict) -> None:
    dcfg  = cfg["data"]
    mcfg  = cfg["model"]
    ocfg  = cfg["output"]
    pcfg  = cfg["preprocessing"]

    feature_col = pcfg["primary_feature_column"]

    # ── load processed data ───────────────────────────────────────────────────
    logger.info("Loading processed data …")
    df = pd.read_csv(dcfg["processed_output_path"], sep="\t")

    X = df[feature_col].fillna("").astype(str)
    y = df["Label"]

    # ── train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=mcfg["test_size"],
        random_state=mcfg["random_state"],
        shuffle=True,
    )
    logger.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # ── tokenise for Word2Vec ─────────────────────────────────────────────────
    X_train_tok = [nltk.word_tokenize(doc) for doc in X_train]
    X_test_tok  = [nltk.word_tokenize(doc) for doc in X_test]

    # ── TF-IDF vectorisation ──────────────────────────────────────────────────
    logger.info("Fitting TF-IDF vectoriser …")
    w2v_cfg = mcfg["word2vec"]
    tfidf = TfidfVectorizer(use_idf=cfg["model"]["tfidf"]["use_idf"])
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    # ── Word2Vec vectorisation ────────────────────────────────────────────────
    logger.info("Training Word2Vec model …")
    w2v_model = Word2Vec(
        sentences=X_train_tok,
        vector_size=w2v_cfg["vector_size"],
        window=w2v_cfg["window"],
        min_count=w2v_cfg["min_count"],
        workers=w2v_cfg["workers"],
        seed=w2v_cfg["seed"],
    )
    # build word → vector lookup dict
    w2v_lookup = dict(zip(w2v_model.wv.index_to_key, w2v_model.wv.vectors))

    mean_vec = MeanEmbeddingVectorizer(w2v_lookup)
    X_train_w2v = mean_vec.transform(X_train_tok)
    X_test_w2v  = mean_vec.transform(X_test_tok)

    # ── train models ──────────────────────────────────────────────────────────
    lr_cfg = mcfg["logistic_regression"]
    results = []

    # 1. LR + TF-IDF
    logger.info("Training LR + TF-IDF …")
    lr_tfidf = LogisticRegression(
        solver=lr_cfg["solver"], C=lr_cfg["C"], penalty=lr_cfg["penalty"]
    )
    lr_tfidf.fit(X_train_tfidf, y_train)
    results.append(_evaluate(
        "LR + TF-IDF",
        y_test,
        lr_tfidf.predict(X_test_tfidf),
        lr_tfidf.predict_proba(X_test_tfidf)[:, 1],
    ))

    # 2. LR + Word2Vec
    logger.info("Training LR + Word2Vec …")
    lr_w2v = LogisticRegression(
        solver=lr_cfg["solver"], C=lr_cfg["C"], penalty=lr_cfg["penalty"]
    )
    lr_w2v.fit(X_train_w2v, y_train)
    results.append(_evaluate(
        "LR + Word2Vec",
        y_test,
        lr_w2v.predict(X_test_w2v),
        lr_w2v.predict_proba(X_test_w2v)[:, 1],
    ))

    # 3. Naïve Bayes + TF-IDF
    logger.info("Training Naïve Bayes + TF-IDF …")
    nb_tfidf = MultinomialNB()
    nb_tfidf.fit(X_train_tfidf, y_train)
    results.append(_evaluate(
        "NB + TF-IDF",
        y_test,
        nb_tfidf.predict(X_test_tfidf),
        nb_tfidf.predict_proba(X_test_tfidf)[:, 1],
    ))

    # ── persist artefacts ─────────────────────────────────────────────────────
    models_dir = ocfg["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(tfidf,     os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(lr_tfidf,  os.path.join(models_dir, "lr_tfidf.joblib"))
    joblib.dump(lr_w2v,    os.path.join(models_dir, "lr_w2v.joblib"))
    joblib.dump(nb_tfidf,  os.path.join(models_dir, "nb_tfidf.joblib"))
    joblib.dump(mean_vec,  os.path.join(models_dir, "mean_embedding_vectorizer.joblib"))
    w2v_model.save(os.path.join(models_dir, "word2vec.model"))
    logger.info("All model artefacts saved → %s", models_dir)

    _plot_roc(results, ocfg["plots_dir"])
    _save_metrics(results, ocfg["metrics_file"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train text classification models.")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config YAML"
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    main(cfg)
