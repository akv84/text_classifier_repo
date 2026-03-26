"""
predict.py
──────────
Apply a saved model to new (unlabelled) data and write predictions to disk.

Usage
-----
# Use the default model (LR + TF-IDF) on a new TSV:
python src/predict.py --config config/config.yaml --input data/new_records.tsv

# Choose a different model:
python src/predict.py --config config/config.yaml \
                      --input data/new_records.tsv \
                      --model lr_w2v          # one of: lr_tfidf | lr_w2v | nb_tfidf

Input file requirements
-----------------------
TSV with at minimum the column specified by
preprocessing.primary_feature_column (default: clean_text_Abstract).

If the raw 'Abstract' column is present instead, pass --raw and the
script will apply the full NLP preprocessing pipeline automatically.
"""

import argparse
import logging
import os

import joblib
import nltk
import pandas as pd
import yaml

# reuse the preprocessing pipeline from preprocess.py
from preprocess import full_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

nltk.download("punkt", quiet=True)

SUPPORTED_MODELS = {
    "lr_tfidf": ("lr_tfidf.joblib",  "tfidf_vectorizer.joblib",  "tfidf"),
    "nb_tfidf": ("nb_tfidf.joblib",  "tfidf_vectorizer.joblib",  "tfidf"),
    "lr_w2v":   ("lr_w2v.joblib",    "word2vec.model",            "w2v"),
}


def load_artefacts(models_dir: str, model_key: str):
    """Load classifier + appropriate vectoriser from disk."""
    clf_file, vec_file, vec_type = SUPPORTED_MODELS[model_key]

    clf = joblib.load(os.path.join(models_dir, clf_file))
    logger.info("Loaded classifier: %s", clf_file)

    if vec_type == "tfidf":
        vectorizer = joblib.load(os.path.join(models_dir, vec_file))
        mean_vec   = None
        logger.info("Loaded vectoriser: %s (TF-IDF)", vec_file)
    else:  # w2v
        from gensim.models import Word2Vec as _W2V
        w2v_model  = _W2V.load(os.path.join(models_dir, vec_file))
        mean_vec   = joblib.load(
            os.path.join(models_dir, "mean_embedding_vectorizer.joblib")
        )
        vectorizer = mean_vec
        logger.info("Loaded vectoriser: %s (Word2Vec mean-pooling)", vec_file)

    return clf, vectorizer, vec_type


def vectorize(texts: pd.Series, vectorizer, vec_type: str):
    """Convert clean text to feature matrix using the saved vectoriser."""
    if vec_type == "tfidf":
        return vectorizer.transform(texts)
    # w2v: tokenise first, then mean-pool
    tokenized = [nltk.word_tokenize(doc) for doc in texts]
    return vectorizer.transform(tokenized)


def main(cfg: dict, input_path: str, model_key: str, raw: bool) -> None:
    ocfg  = cfg["output"]
    pcfg  = cfg["preprocessing"]
    feature_col = pcfg["primary_feature_column"]

    # ── load new data ─────────────────────────────────────────────────────────
    logger.info("Loading input file: %s", input_path)
    df = pd.read_csv(input_path, sep="\t")

    # ── optionally preprocess raw text ────────────────────────────────────────
    if raw:
        raw_col = feature_col.replace("clean_text_", "")  # e.g. "Abstract"
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

    texts = df[feature_col].fillna("").astype(str)

    # ── load artefacts and vectorise ──────────────────────────────────────────
    clf, vectorizer, vec_type = load_artefacts(ocfg["models_dir"], model_key)
    X = vectorize(texts, vectorizer, vec_type)

    # ── predict ───────────────────────────────────────────────────────────────
    df["predicted_label"] = clf.predict(X)
    df["predicted_prob"]  = clf.predict_proba(X)[:, 1]

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(ocfg["predictions_dir"], exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(ocfg["predictions_dir"], f"{base}_predictions_{model_key}.tsv")
    df.to_csv(out_path, sep="\t", index=False)
    logger.info("Predictions saved → %s", out_path)

    # quick summary
    vc = df["predicted_label"].value_counts()
    logger.info("Prediction summary:\n%s", vc.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply a trained text classifier to new data."
    )
    parser.add_argument("--config",  default="config/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--input",   required=True,
                        help="Path to new data TSV file")
    parser.add_argument("--model",   default="lr_tfidf",
                        choices=list(SUPPORTED_MODELS.keys()),
                        help="Which saved model to use (default: lr_tfidf)")
    parser.add_argument("--raw",     action="store_true",
                        help="Input contains raw text; apply NLP pipeline before prediction")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    main(cfg, args.input, args.model, args.raw)
