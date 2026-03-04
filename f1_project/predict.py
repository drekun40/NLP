"""
predict.py — F1 Radio NLP · Pace Predictor (Version 2 — Binary)

Usage:
    python3 predict.py "Box box, pit for fresh tyres"
    python3 predict.py "Push harder!" --tyre-age 0.8 --position 0.4
    python3 predict.py --interactive
"""
import os, sys, re, pickle, warnings, argparse
import numpy as np
import spacy
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack, csr_matrix

nlp      = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()

F1_FILLER = {
    "copy", "understood", "roger", "okay", "ok", "yeah", "yes", "yep",
    "no", "uh", "um", "er", "ah", "hmm", "right", "alright",
}
F1_DOMAIN = {
    "drs", "vsc", "sc", "pit", "box", "tyre", "tire", "compound",
    "soft", "medium", "hard", "undercut", "overcut", "delta", "safety",
    "car", "strat", "strategy", "push", "gap", "interval", "lap", "sector",
}


# ── Text preprocessing ────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text: str) -> str:
    doc = nlp(_clean(text).lower())
    tokens = []
    for tok in doc:
        if tok.is_punct or tok.like_num or tok.is_space:
            continue
        lem = tok.lemma_.lower()
        if lem in ENGLISH_STOP_WORDS and lem not in F1_DOMAIN:
            continue
        if lem in F1_FILLER or len(lem) < 2:
            continue
        tokens.append(lem)
    return " ".join(tokens)


# ── Load model ────────────────────────────────────────────────────────────────

_dir      = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_dir, "sentiment_model.pkl"), "rb") as f:
    _payload = pickle.load(f)

_model     = _payload["lgb"]
_tfidf     = _payload["tfidf"]
_num_cols  = _payload["num_cols"]
_threshold = _payload.get("threshold", 0.54)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(message: str, context: dict = None) -> tuple[int, float]:
    """
    Predict whether a team radio message will affect driver pace.

    Parameters
    ----------
    message : str
        Raw team radio transcript.
    context : dict, optional
        Race metadata (all values normalised 0–1):
          tyre_age_norm         — 0=new tyres, 1=heavily worn
          position_norm         — 0=P1, 1=P20
          driver_affected_rate  — driver's historical pace-change rate

    Returns
    -------
    (prediction, probability)
        prediction : int   — 1=AFFECTED, 0=STABLE
        probability: float — P(AFFECTED)
    """
    sentiment  = analyzer.polarity_scores(message)["compound"]
    word_count = len(message.split())

    features = {
        "sentiment_score":        sentiment,
        "transcript_word_count":  word_count,
        "lap_duration_norm":      0.5,
        "tyre_age_norm":          0.5,
        "position_norm":          0.5,
        "air_temperature_norm":   0.5,
        "track_temperature_norm": 0.5,
        "wind_speed_norm":        0.3,
        "driver_affected_rate":   0.5,
        "circuit_volatility":     0.0,
        "tyre_sentiment":         0.5 * sentiment,
        "msg_short":              int(word_count < 5),
        "msg_long":               int(word_count > 15),
        "negative_msg":           int(sentiment < -0.2),
        "positive_msg":           int(sentiment > 0.3),
    }
    if context:
        features.update(context)
        features["tyre_sentiment"] = features["tyre_age_norm"] * sentiment

    X_text  = _tfidf.transform([preprocess(message)])
    X_num   = csr_matrix([[features.get(c, 0.0) for c in _num_cols]])
    p       = float(_model.predict_proba(hstack([X_text, X_num]))[0, 1])
    pred    = 1 if p >= _threshold else 0
    conf    = p if pred == 1 else 1.0 - p
    label   = "AFFECTED" if pred == 1 else "STABLE"
    ctx_tag = " (with context)" if context else ""

    print(f'  "{message}"')
    print(f"  → {label}  ({conf:.1%} confidence{ctx_tag})\n")
    return pred, p


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="F1 Radio NLP — Pace Predictor v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("message", nargs="*", help="Radio message text")
    parser.add_argument("--tyre-age",    type=float, metavar="0-1",
                        help="Tyre wear (0=new, 1=worn)")
    parser.add_argument("--position",    type=float, metavar="0-1",
                        help="Track position (0=P1, 1=P20)")
    parser.add_argument("--driver-rate", type=float, metavar="0-1",
                        help="Driver historical affected rate")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter message and context interactively")
    args = parser.parse_args()

    ctx = {}
    if args.tyre_age    is not None: ctx["tyre_age_norm"]       = args.tyre_age
    if args.position    is not None: ctx["position_norm"]        = args.position
    if args.driver_rate is not None: ctx["driver_affected_rate"] = args.driver_rate

    print(f"\nF1 Pace Predictor  [threshold={_threshold:.2f}]\n")

    if args.interactive:
        msg = input("Radio message : ").strip()
        ta  = input("Tyre age 0–1  [↵ skip]: ").strip()
        pos = input("Position 0–1  [↵ skip]: ").strip()
        if ta:  ctx["tyre_age_norm"]  = float(ta)
        if pos: ctx["position_norm"]  = float(pos)
        predict(msg, ctx or None)
    elif args.message:
        predict(" ".join(args.message), ctx or None)
    else:
        parser.print_help()
