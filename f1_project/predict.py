"""
predict.py — Use the trained F1 radio NLP model to make predictions.

Usage:
    python3 predict.py "Box box, come in now, we're going to pit"
    python3 predict.py "Push push push, gap is closing, push!"
    python3 predict.py "Copy that, understood"
"""
import sys
import pickle
import re
import warnings
import spacy
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack, csr_matrix
import numpy as np

nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()

F1_FILLER_WORDS = {
    "copy", "understood", "roger", "okay", "ok", "yeah", "yes", "yep",
    "no", "uh", "um", "er", "ah", "hmm", "right", "alright"
}
F1_DOMAIN_WORDS = {
    "drs", "vsc", "sc", "pit", "box", "tyre", "tire", "compound",
    "soft", "medium", "hard", "undercut", "overcut", "delta", "safety",
    "car", "strat", "strategy", "push", "gap", "interval", "lap", "sector"
}

def clean_text(text):
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()

def preprocess(text):
    text = clean_text(text).lower()
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_punct or token.like_num or token.is_space:
            continue
        lemma = token.lemma_.lower()
        if lemma in ENGLISH_STOP_WORDS and lemma not in F1_DOMAIN_WORDS:
            continue
        if lemma in F1_FILLER_WORDS or len(lemma) < 2:
            continue
        tokens.append(lemma)
    return " ".join(tokens)

# Load model
with open("sentiment_model.pkl", "rb") as f:
    payload = pickle.load(f)

model  = payload["lgb"]
tfidf  = payload["tfidf"]
num_cols = payload["num_cols"]

LABEL_EMOJI = {"DOWN": "📉 DOWN (driver likely gets FASTER)", 
               "UP":   "📈 UP   (driver likely gets SLOWER)",
               "NEUTRAL": "➡️  NEUTRAL (no significant change)"}

def predict(radio_message: str):
    clean = preprocess(radio_message)
    sentiment = analyzer.polarity_scores(radio_message)["compound"]
    word_count = len(radio_message.split())

    X_text = tfidf.transform([clean])

    # Build numerical features (zeros for missing race context)
    num_values = []
    defaults = {
        "sentiment_score": sentiment,
        "transcript_word_count": word_count,
        "lap_duration_norm": 0.5,
        "tyre_age_norm": 0.5,
        "position_norm": 0.5,
        "air_temperature_norm": 0.5,
        "track_temperature_norm": 0.5,
        "wind_speed_norm": 0.3,
        "driver_down_rate": 0.4,
        "circuit_avg_delta": 0.0,
        "tyre_sentiment": 0.5 * sentiment,
        "msg_short": int(word_count < 5),
        "msg_long": int(word_count > 15),
        "negative_msg": int(sentiment < -0.2),
        "positive_msg": int(sentiment > 0.3),
    }
    for col in num_cols:
        num_values.append(defaults.get(col, 0.0))

    X_num = csr_matrix([num_values])
    X = hstack([X_text, X_num])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    classes = model.classes_

    print(f"\n{'='*55}")
    print(f" Radio message : \"{radio_message}\"")
    print(f" Clean tokens  : \"{clean}\"")
    print(f" Sentiment     : {sentiment:+.3f}")
    print(f"{'='*55}")
    print(f" Prediction    : {LABEL_EMOJI[pred]}")
    print(f"\n Probabilities:")
    for cls, prob in sorted(zip(classes, proba), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"   {cls:<8} {bar:<30} {prob:.1%}")
    print()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        predict(message)
    else:
        # Demo with a few examples
        examples = [
            "Box box, come in now, put the soft tyres on",
            "Push push push! Gap is only 1.2, we need to attack!",
            "Copy that, understood, we'll check the data",
            "Something is wrong with the engine, I have no power",
            "Valtteri, it's James. We need you to hold position behind Lewis.",
        ]
        for msg in examples:
            predict(msg)
