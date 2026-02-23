"""
preprocess.py — improved preprocessing for F1 radio NLP model

Improvements over basic version:
1. Lemmatization — "pushing" → "push", "tyres" → "tyre"
2. F1 filler word removal — removes radio noise words like "copy", "understood", "roger"
3. VADER sentiment score — polarity of the message as a numerical feature
4. Transcript noise cleaning — removes Whisper artifacts like "[inaudible]", repeated words
5. Keeps F1-specific technical words — "drs", "vsc", "undercut", "pit", "safety car"
6. Normalizes numerical features — position, tyre_age, lap_duration
"""
import re
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()

# F1-specific filler words that add no predictive signal
F1_FILLER_WORDS = {
    "copy", "understood", "roger", "okay", "ok", "yeah", "yes", "yep",
    "no", "uh", "um", "er", "ah", "hmm", "right", "alright", "sure",
    "hello", "hi", "bye", "thank", "thanks", "please", "sorry"
}

# F1 technical words — KEEP these even if they'd be filtered by stopwords
F1_DOMAIN_WORDS = {
    "drs", "vsc", "sc", "pit", "box", "tyre", "tire", "compound",
    "soft", "medium", "hard", "undercut", "overcut", "delta",
    "safety", "car", "strat", "strategy", "push", "gap", "interval",
    "lap", "sector", "overtake", "brake", "throttle", "downshift",
    "upshift", "engine", "front", "rear", "wing", "fuel", "deg",
    "degradation", "pace", "stint", "position", "radio", "team"
}

def clean_transcript(text):
    """Remove Whisper artifacts and noise from transcription."""
    # Remove brackets like [inaudible], [noise], [music]
    text = re.sub(r"\[.*?\]", "", text)
    # Remove excessive punctuation
    text = re.sub(r"[^\w\s']", " ", text)
    # Collapse repeated words (e.g. "push push push" → "push")
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)
    # Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text(text):
    """Full preprocessing pipeline with lemmatization."""
    text = clean_transcript(text)
    text = text.lower()
    doc = nlp(text)

    tokens = []
    for token in doc:
        if token.is_punct or token.like_num or token.is_space:
            continue
        lemma = token.lemma_.lower()
        # Skip generic stop words UNLESS it's a domain-important word
        if lemma in ENGLISH_STOP_WORDS and lemma not in F1_DOMAIN_WORDS:
            continue
        # Remove F1 filler words
        if lemma in F1_FILLER_WORDS:
            continue
        # Skip very short tokens (1 char)
        if len(lemma) < 2:
            continue
        tokens.append(lemma)

    return tokens

# --- Load data ---
df = pd.read_csv("raw_f1_dataset.csv")
df = df[df["transcript"].notna() & df["label"].notna()].copy()
df = df[df["transcript"].str.strip() != ""].copy()
print(f"{len(df)} rows with transcript + label")

# --- Text preprocessing ---
df["tokens"] = df["transcript"].apply(preprocess_text)
df["clean_text"] = df["tokens"].apply(lambda tokens: " ".join(tokens))

# Drop rows where clean_text is empty after preprocessing (pure noise transcripts)
df = df[df["clean_text"].str.strip() != ""].copy()
print(f"{len(df)} rows after removing empty post-clean transcripts")

# --- VADER sentiment score ---
def get_sentiment(text):
    scores = analyzer.polarity_scores(str(text))
    return scores["compound"]  # -1 (very negative) to +1 (very positive)

df["sentiment_score"] = df["transcript"].apply(get_sentiment)

# --- Transcript length as feature ---
df["transcript_word_count"] = df["transcript"].apply(lambda t: len(str(t).split()))

# --- Binary affected label ---
df["affected"] = df["label"].apply(lambda x: 1 if x in ["UP", "DOWN"] else 0)

# --- Normalize numerical features ---
num_cols = ["lap_duration", "tyre_age", "position", "air_temperature",
            "track_temperature", "wind_speed"]
available_num_cols = [c for c in num_cols if c in df.columns]

scaler = MinMaxScaler()
df_num = df[available_num_cols].copy()
df_num_filled = df_num.fillna(df_num.median())
df[[f"{c}_norm" for c in available_num_cols]] = scaler.fit_transform(df_num_filled)

# --- Save ---
df.to_csv("processed_f1_dataset.csv", index=False)
print(f"\nSaved processed_f1_dataset.csv — {len(df)} rows")

print("\nSample transcripts:")
print(df[["name_acronym", "transcript", "clean_text", "sentiment_score", "label"]].head(5).to_string())

print("\n3-class label distribution:")
print(df["label"].value_counts())

print("\nBinary affected distribution:")
print(df["affected"].value_counts())

print("\nSentiment score stats:")
print(df["sentiment_score"].describe().round(3))

print(f"\nFeatures available for model:")
feature_cols = ["clean_text", "sentiment_score", "transcript_word_count"] + \
               [f"{c}_norm" for c in available_num_cols]
print(feature_cols)
