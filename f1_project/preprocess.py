"""
preprocess.py — F1 Radio NLP · Data Preprocessing Pipeline

Steps:
  1. Load raw_f1_dataset.csv
  2. Clean transcripts  (Whisper artifacts, repeated words, punctuation)
  3. Lemmatize with spaCy, removing stop-words & filler phrases
  4. Compute VADER sentiment score
  5. Derive binary label  UP/DOWN → AFFECTED (1),  NEUTRAL → STABLE (0)
  6. Normalize numerical race features with MinMaxScaler
  7. Save processed_f1_dataset.csv
"""
import re
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nlp      = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()

# Radio filler words that carry no predictive signal
F1_FILLER = {
    "copy", "understood", "roger", "okay", "ok", "yeah", "yes", "yep",
    "no", "uh", "um", "er", "ah", "hmm", "right", "alright", "sure",
    "hello", "hi", "bye", "thank", "thanks", "please", "sorry",
}

# F1 technical terms — keep these even when they match generic stop-words
F1_DOMAIN = {
    "drs", "vsc", "sc", "pit", "box", "tyre", "tire", "compound",
    "soft", "medium", "hard", "undercut", "overcut", "delta",
    "safety", "car", "strat", "strategy", "push", "gap", "interval",
    "lap", "sector", "overtake", "brake", "throttle", "downshift",
    "upshift", "engine", "front", "rear", "wing", "fuel",
    "degradation", "pace", "stint", "position", "radio", "team",
}


# ── Text helpers ──────────────────────────────────────────────────────────────

def clean_transcript(text: str) -> str:
    """Remove Whisper artifacts and normalise whitespace."""
    text = re.sub(r"\[.*?\]", "", text)           # [inaudible], [noise] …
    text = re.sub(r"[^\w\s']", " ", text)         # punctuation
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)  # repeated words
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text: str) -> list[str]:
    """Lemmatize and filter tokens."""
    doc = nlp(clean_transcript(text).lower())
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
    return tokens


# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv("raw_f1_dataset.csv")
df = df[df["transcript"].notna() & df["label"].notna()].copy()
df = df[df["transcript"].str.strip() != ""].copy()

# ── Preprocessing ─────────────────────────────────────────────────────────────

df["clean_text"] = df["transcript"].apply(
    lambda t: " ".join(preprocess_text(t))
)
df = df[df["clean_text"].str.strip() != ""].copy()

df["sentiment_score"]      = df["transcript"].apply(
    lambda t: analyzer.polarity_scores(str(t))["compound"]
)
df["transcript_word_count"] = df["transcript"].apply(
    lambda t: len(str(t).split())
)

# Binary label: pace changes (UP or DOWN) → 1, no change → 0
df["affected"] = df["label"].apply(lambda x: 1 if x in ["UP", "DOWN"] else 0)

# ── Normalize numerical features ──────────────────────────────────────────────

NUM_COLS = ["lap_duration", "tyre_age", "position",
            "air_temperature", "track_temperature", "wind_speed"]
available = [c for c in NUM_COLS if c in df.columns]

scaler    = MinMaxScaler()
filled    = df[available].fillna(df[available].median())
df[[f"{c}_norm" for c in available]] = scaler.fit_transform(filled)

# ── Save ──────────────────────────────────────────────────────────────────────

df.to_csv("processed_f1_dataset.csv", index=False)

print(f"Saved processed_f1_dataset.csv")
print(f"  Rows    : {len(df):,}")
print(f"  AFFECTED: {df['affected'].sum():,}  |  STABLE: {(df['affected']==0).sum():,}")
print(f"  Columns : {len(df.columns)}")
