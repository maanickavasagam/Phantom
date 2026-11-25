"""ECHO learned scam classifier (Week 4).

A small, explainable text classifier that complements the rules: TF-IDF word
1-2 grams + logistic regression, trained ONLY on the official train halves of
two public datasets (BothBosu single-agent and multi-agent scam dialogues,
Apache-2.0). The rules catch explicit asks (OTP, gift cards...); the
classifier picks up softer scam scripts ("your subscription renewed... are you
in front of your computer?") that have no single keyword.

Guards against learning dataset quirks instead of scams:
- speaker labels removed; money/currency masked ("$500", "Rs 50,000" -> <amount>)
  so "dollars" can't become a scam feature that fails on Indian calls;
- digits masked; min_df so one-off names don't matter;
- the decision threshold is chosen from 5-fold cross-validation on the train
  data only, capped at ~2% false positives.

Every score is explained by the phrases in the transcript that pushed it up.

    python -m phantom.echo.scam_classifier train --root data/raw/bothbosu
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "scam_clf.joblib"
META_PATH = MODEL_DIR / "scam_clf.json"
MAX_OOF_FPR = 0.02

_SPEAKER = re.compile(r"\b(?:suspect|innocent|person [ab]|caller|receiver)\s*:", re.I)
_MONEY = re.compile(r"(?:₹|rs\.?|inr|\$|usd|£|€)\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?"
                    r"(?:rupees?|dollars?|bucks|lakhs?|crores?|usd|inr)\b", re.I)


def normalize(text: str) -> str:
    t = _SPEAKER.sub(" ", text)
    t = _MONEY.sub(" <amount> ", t)
    t = re.sub(r"\d", "0", t)
    return re.sub(r"\s+", " ", t).strip().lower()


@dataclass
class Score:
    probability: float
    threshold: float
    top_phrases: list[tuple[str, float]]  # phrase, contribution

    @property
    def suspicious(self) -> bool:
        return self.probability >= self.threshold


def _pipeline(c: float):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000,
                                  sublinear_tf=True, token_pattern=r"(?u)<amount>|\b\w[\w']+\b")),
        ("lr", LogisticRegression(C=c, max_iter=2000, class_weight="balanced")),
    ])


LOCAL_WEIGHT = 20.0  # 42 own dev calls vs 2,560 public dialogues


def _local_dev_calls(calls_json: Path | None) -> list[tuple[str, int]]:
    """Transcripts of our own scripted DEV calls (clean audio). The public data has
    no ordinary Indian money talk (bill reminders, UPI between friends), and
    without these the classifier flagged half of our genuine calls. TEST calls
    are never used here."""
    if not calls_json or not calls_json.exists():
        return []
    data = json.loads(calls_json.read_text())
    assert data.get("split") == "dev", "only the dev split may be used for training"
    return [(r["hypothesis"], int(r["label"] == "scam")) for r in data["rows"]
            if r["condition"] == "clean" and r["hypothesis"]]


def _fit(c: float, texts, y, w):
    return _pipeline(c).fit(texts, y, lr__sample_weight=w)


def train(root: Path, dev_calls: Path | None = None) -> dict:
    """Fit on the public train halves (+ our dev calls); choose C and the
    threshold by 5-fold cross-validation on that training data only."""
    import joblib
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    from phantom.echo.eval_public import _rows

    texts, labels, weights, local = [], [], [], []
    for name in ("single_agent", "multi_agent"):
        for text, label, _ in _rows(root, name, "train"):
            texts.append(normalize(text))
            labels.append(label)
            weights.append(1.0)
            local.append(False)
    for text, label in _local_dev_calls(dev_calls):
        texts.append(normalize(text))
        labels.append(label)
        weights.append(LOCAL_WEIGHT)
        local.append(True)
    texts, y, w, local = np.array(texts, dtype=object), np.array(labels), np.array(weights), \
        np.array(local)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    strata = y * 2 + local  # keep local calls spread across folds
    best = None
    for c in (0.3, 1.0, 3.0, 10.0):
        oof = np.zeros(len(y))
        for tr_i, te_i in cv.split(texts, strata):
            oof[te_i] = _fit(c, list(texts[tr_i]), y[tr_i], w[tr_i]).predict_proba(
                list(texts[te_i]))[:, 1]
        # Highest-recall threshold whose out-of-fold false-positive rate is <= MAX_OOF_FPR,
        # computed separately on public and local genuine calls (both must hold).
        thr = 0.0
        for group in (~local, local):
            neg = np.sort(oof[(y == 0) & group])
            if len(neg):
                thr = max(thr, float(neg[int(np.ceil(len(neg) * (1 - MAX_OOF_FPR))) - 1]) + 1e-6)
        rec = float((oof[y == 1] >= thr).mean())
        fpr = float((oof[y == 0] >= thr).mean())
        if best is None or rec > best["cv_recall"]:
            best = {"C": c, "threshold": round(thr, 4), "cv_recall": round(rec, 3),
                    "cv_fpr": round(fpr, 3),
                    "cv_fpr_local": round(float((oof[(y == 0) & local] >= thr).mean()), 3)
                    if local.any() else None}
    model = _fit(best["C"], list(texts), y, w)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    meta = {**best, "n_train": len(texts), "n_scam": int(y.sum()), "n_local_dev": int(local.sum()),
            "trained_on": "BothBosu single-agent + multi-agent official TRAIN halves"
                          + (" plus our scripted DEV call transcripts (weight 20)"
                             if local.any() else "") + "; no test data",
            "features": "tfidf word 1-2 grams on normalized text "
                        "(speakers removed, money/digits masked)",
            "max_oof_fpr": MAX_OOF_FPR}
    META_PATH.write_text(json.dumps(meta, indent=2))
    return meta


@lru_cache(maxsize=1)
def _load():
    import joblib

    if not MODEL_PATH.exists():
        return None, None
    meta = json.loads(META_PATH.read_text())
    return joblib.load(MODEL_PATH), meta


def available() -> bool:
    return _load()[0] is not None


def score(text: str, top_k: int = 6) -> Score | None:
    """Scam probability for a transcript, with the phrases that raised it most."""
    model, meta = _load()
    if model is None:
        return None
    norm = normalize(text)
    prob = float(model.predict_proba([norm])[0, 1])
    vec, lr = model.named_steps["tfidf"], model.named_steps["lr"]
    x = vec.transform([norm])
    names = vec.get_feature_names_out()
    contrib = [(names[j], float(x[0, j] * lr.coef_[0, j])) for j in x.nonzero()[1]]
    top = sorted((c for c in contrib if c[1] > 0), key=lambda c: -c[1])[:top_k]
    return Score(round(prob, 3), meta["threshold"], [(p, round(w, 3)) for p, w in top])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--root", default="data/raw/bothbosu")
    t.add_argument("--dev-calls", default="docs/echo_eval_calls_dev.json",
                   help="our scripted dev-call results (transcripts); '' to skip")
    a = ap.parse_args()
    print(json.dumps(train(Path(a.root), Path(a.dev_calls) if a.dev_calls else None), indent=2))
