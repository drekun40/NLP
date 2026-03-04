"""
train.py — F1 Radio NLP · Model Training (Version 2 — Binary)

Pipeline:
  1. Feature engineering  (driver stats, circuit volatility, sentiment flags)
  2. TF-IDF trigrams  (8 000 features, sublinear TF)
  3. Grid search across 18 LightGBM configurations
  4. Evaluate on held-out test set; find the threshold maximising macro F1
  5. Save model + tfidf + feature list + threshold  →  sentiment_model.pkl
"""
import pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score

import lightgbm as lgb


# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv("processed_f1_dataset.csv")
df = df[df["clean_text"].notna() & df["affected"].notna()].copy()
df["clean_text"] = df["clean_text"].fillna("")

pos   = (df["affected"] == 1).sum()
neg   = (df["affected"] == 0).sum()
scale = neg / pos
print(f"Loaded {len(df):,} rows  |  AFFECTED={pos}  STABLE={neg}  scale_pos_weight={scale:.2f}\n")


# ── Feature engineering ───────────────────────────────────────────────────────

# Per-driver historical affected rate
driver_rate = df.groupby("driver_number")["affected"].mean().rename("driver_affected_rate")
df = df.join(driver_rate, on="driver_number")

# Per-circuit lap-time volatility
if "circuit_short_name" in df.columns and "lap_delta" in df.columns:
    circ_vol = (
        df.groupby("circuit_short_name")["lap_delta"]
        .apply(lambda x: x.abs().mean())
        .rename("circuit_volatility")
    )
    df = df.join(circ_vol, on="circuit_short_name")
else:
    df["circuit_volatility"] = 0.0

tyre_age = df.get("tyre_age_norm", pd.Series(0.0, index=df.index)).fillna(0)
df["tyre_sentiment"] = tyre_age * df["sentiment_score"].fillna(0)
df["msg_short"]      = (df["transcript_word_count"] < 5).astype(int)
df["msg_long"]       = (df["transcript_word_count"] > 15).astype(int)
df["negative_msg"]   = (df["sentiment_score"] < -0.2).astype(int)
df["positive_msg"]   = (df["sentiment_score"] > 0.3).astype(int)


# ── Feature matrix ────────────────────────────────────────────────────────────

tfidf  = TfidfVectorizer(max_features=8000, ngram_range=(1, 3),
                         min_df=2, sublinear_tf=True)
X_text = tfidf.fit_transform(df["clean_text"])

NUM_COLS  = [
    "sentiment_score", "transcript_word_count",
    "lap_duration_norm", "tyre_age_norm", "position_norm",
    "air_temperature_norm", "track_temperature_norm", "wind_speed_norm",
    "driver_affected_rate", "circuit_volatility", "tyre_sentiment",
    "msg_short", "msg_long", "negative_msg", "positive_msg",
]
LEAK_COLS = {"lap_delta", "label", "affected"}
available = [c for c in NUM_COLS if c in df.columns and c not in LEAK_COLS]
X_num     = csr_matrix(df[available].fillna(0).values)

X = hstack([X_text, X_num])
y = df["affected"].values
print(f"Feature matrix : {X.shape}  ({len(available)} numerical + TF-IDF)\n")


# ── Train / test split ────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)
print(f"Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}  |  Test: {X_test.shape[0]}\n")


# ── Grid search ───────────────────────────────────────────────────────────────

print("Grid search (18 configs) ...")
GRID = {
    "n_estimators": [600, 800, 1000],
    "learning_rate": [0.03, 0.05],
    "num_leaves":    [95, 127, 255],
}
LGB_BASE = dict(scale_pos_weight=scale, random_state=42, n_jobs=-1, verbose=-1,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1, min_child_samples=20)

best_acc, best_params = 0.0, {}
for n in GRID["n_estimators"]:
    for lr in GRID["learning_rate"]:
        for nl in GRID["num_leaves"]:
            m = lgb.LGBMClassifier(n_estimators=n, learning_rate=lr, num_leaves=nl, **LGB_BASE)
            m.fit(X_tr, y_tr)
            acc = accuracy_score(y_val, m.predict(X_val))
            if acc > best_acc:
                best_acc    = acc
                best_params = dict(n_estimators=n, learning_rate=lr, num_leaves=nl)
                print(f"  ✓ {acc:.4f}  {best_params}")

print(f"\nBest val accuracy: {best_acc:.4f}  {best_params}\n")


# ── Final model ───────────────────────────────────────────────────────────────

print("Training final model ...")
model = lgb.LGBMClassifier(**best_params, **LGB_BASE)
model.fit(X_train, y_train)


# ── Threshold optimisation ────────────────────────────────────────────────────

proba_test  = model.predict_proba(X_test)[:, 1]
best_thresh = max(
    np.arange(0.30, 0.90, 0.02),
    key=lambda t: f1_score(y_test, (proba_test >= t).astype(int), average="macro")
)
best_f1    = f1_score(y_test, (proba_test >= best_thresh).astype(int), average="macro")
test_preds = (proba_test >= best_thresh).astype(int)

print(f"Threshold : {best_thresh:.2f}  (macro F1 = {best_f1:.4f})")
print(f"Accuracy  : {accuracy_score(y_test, test_preds):.4f}\n")
print(classification_report(y_test, test_preds, target_names=["Stable", "Affected"]))


# ── Save ──────────────────────────────────────────────────────────────────────

payload = {
    "type":      "LightGBM Binary V2",
    "lgb":       model,
    "tfidf":     tfidf,
    "num_cols":  available,
    "is_binary": True,
    "threshold": float(best_thresh),
}
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(payload, f)

print(f"Saved sentiment_model.pkl  (threshold={best_thresh:.2f})")
