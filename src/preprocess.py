"""
preprocess.py
─────────────
Merge positive / negative PMID datasets, remove overlapping PMIDs,
apply full NLP preprocessing pipeline, and save the processed data.

Usage
-----
python src/preprocess.py --config config/config.yaml
"""

import argparse
import glob
import logging
import os
import re
import string

import nltk
import pandas as pd
import yaml
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── download NLTK data once ───────────────────────────────────────────────────
for pkg in ("punkt", "averaged_perceptron_tagger", "wordnet", "stopwords"):
    nltk.download(pkg, quiet=True)

_wl = WordNetLemmatizer()
_stop = set(stopwords.words("english"))


# ── NLP helpers ───────────────────────────────────────────────────────────────

def _get_wordnet_pos(tag: str) -> str:
    """Map Penn-Treebank POS tag to WordNet POS constant."""
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # default


def preprocess_text(text: str) -> str:
    """Lowercase, strip HTML, remove punctuation/digits, collapse whitespace."""
    text = str(text).lower().strip()
    text = re.compile(r"<.*?>").sub("", text)                       # strip HTML
    text = re.compile(r"[%s]" % re.escape(string.punctuation)).sub(" ", text)
    text = re.sub(r"\[\d+\]", " ", text)                            # citation refs
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", " ", text)                                # digits
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    return " ".join(w for w in text.split() if w not in _stop)


def lemmatize(text: str) -> str:
    tagged = nltk.pos_tag(word_tokenize(text))
    return " ".join(
        _wl.lemmatize(word, _get_wordnet_pos(pos)) for word, pos in tagged
    )


def full_pipeline(text: str) -> str:
    """preprocess → remove stopwords → lemmatize."""
    return lemmatize(remove_stopwords(preprocess_text(text)))


# ── data loading ──────────────────────────────────────────────────────────────

def load_negative(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    logger.info("Loading %d negative batch file(s) …", len(files))
    return pd.concat((pd.read_csv(f, sep="\t") for f in files), ignore_index=True)


def load_positive(path: str) -> pd.DataFrame:
    logger.info("Loading positive data from %s …", path)
    return pd.read_csv(path, sep="\t")


# ── main ──────────────────────────────────────────────────────────────────────

def main(cfg: dict) -> None:
    dcfg = cfg["data"]
    pcfg = cfg["preprocessing"]

    neg_df = load_negative(dcfg["negative_data_pattern"])
    pos_df = load_positive(dcfg["positive_data_path"])

    # remove PMIDs that appear in both sets
    overlap = set(neg_df["PMID"]) & set(pos_df["PMID"])
    logger.info("Removing %d overlapping PMIDs …", len(overlap))
    neg_df = neg_df[~neg_df["PMID"].isin(overlap)].copy()
    pos_df = pos_df[~pos_df["PMID"].isin(overlap)].copy()

    neg_df["Label"] = 0
    pos_df["Label"] = 1

    all_df = pd.concat([neg_df, pos_df], ignore_index=True)
    logger.info("Dataset size after merge: %d rows", len(all_df))

    # apply NLP pipeline to each configured text column
    for col in pcfg["text_columns"]:
        if col not in all_df.columns:
            logger.warning("Column '%s' not found — skipping.", col)
            continue
        out_col = f"clean_text_{col}"
        logger.info("Preprocessing column: %s → %s …", col, out_col)
        all_df[out_col] = all_df[col].apply(full_pipeline)

    out_path = dcfg["processed_output_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    all_df.to_csv(out_path, sep="\t", index=False)
    logger.info("Saved processed data → %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess text classification data.")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config YAML"
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    main(cfg)
