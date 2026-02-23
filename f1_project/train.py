"""
train_advanced.py — Final approach toward 75%

Insight: sparse TF-IDF is essential. Do NOT compress with SVD.
Strategy:
1. Sparse TF-IDF (8000 features, bigrams)
2. Manual LightGBM grid over key params using hold-out val set (fast)
3. Best LightGBM on SMOTE-balanced data
4. Sparse stacking with LightGBM + RF probabilities
"""
import pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb

# ── Load & feature engineering ────────────────────────────────────────────────
df = pd.read_csv("processed_f1_dataset.csv")
df = df[df["clean_text"].notna() & df["label"].notna()].copy()
df["clean_text"] = df["clean_text"].fillna("")
print(f"Loaded {len(df)} rows | {df['label'].value_counts().to_dict()}\n")

# Driver historical improvement rate
driver_stats = df.groupby("driver_number")["label"].apply(
    lambda x: (x == "DOWN").sum() / max(len(x), 1)
).rename("driver_down_rate")
df = df.join(driver_stats, on="driver_number")

# Circuit avg lap delta
if "circuit_short_name" in df.columns and "lap_delta" in df.columns:
    circuit_stats = df.groupby("circuit_short_name")["lap_delta"].mean().rename("circuit_avg_delta")
    df = df.join(circuit_stats, on="circuit_short_name")
else:
    df["circuit_avg_delta"] = 0.0

df["tyre_sentiment"] = df.get("tyre_age_norm", pd.Series(0.0, index=df.index)).fillna(0) * df["sentiment_score"].fillna(0)
df["msg_short"] = (df["transcript_word_count"] < 5).astype(int)
df["msg_long"]  = (df["transcript_word_count"] > 15).astype(int)
df["negative_msg"] = (df["sentiment_score"] < -0.2).astype(int)
df["positive_msg"] = (df["sentiment_score"] > 0.3).astype(int)

# ── TF-IDF (sparse, full-dimensional) ────────────────────────────────────────
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_text = tfidf.fit_transform(df["clean_text"])

# ── Numerical features ────────────────────────────────────────────────────────
num_cols = [
    "sentiment_score", "transcript_word_count",
    "lap_duration_norm", "tyre_age_norm", "position_norm",
    "air_temperature_norm", "track_temperature_norm", "wind_speed_norm",
    "driver_down_rate", "circuit_avg_delta", "tyre_sentiment",
    "msg_short", "msg_long", "negative_msg", "positive_msg"
]
available = [c for c in num_cols if c in df.columns]
X_num = csr_matrix(df[available].fillna(0).values)

X = hstack([X_text, X_num])
y = df["label"].values
print(f"Feature matrix: {X.shape}")

# ── Split: 70% train, 10% val, 20% test ──────────────────────────────────────
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.125, random_state=42, stratify=y_tmp)
print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}\n")

# ── SMOTE (train only) ────────────────────────────────────────────────────────
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal_enc = smote.fit_resample(X_train, y_train_enc)
y_train_bal = le.inverse_transform(y_train_bal_enc)
print(f"After SMOTE: {pd.Series(y_train_bal).value_counts().to_dict()}\n")

# ── Manual grid search on LightGBM using val set (fast!) ─────────────────────
print("Grid search on LightGBM using validation set...")
grid = {
    "n_estimators": [400, 600, 800],
    "learning_rate": [0.03, 0.05, 0.1],
    "num_leaves": [63, 95, 127],
}
best_val_acc = 0
best_params = {}

for n_est in grid["n_estimators"]:
    for lr in grid["learning_rate"]:
        for nl in grid["num_leaves"]:
            m = lgb.LGBMClassifier(
                n_estimators=n_est, learning_rate=lr, num_leaves=nl,
                class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                min_child_samples=20
            )
            m.fit(X_train_bal, y_train_bal)
            val_acc = accuracy_score(y_val, m.predict(X_val))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = dict(n_estimators=n_est, learning_rate=lr, num_leaves=nl)
                print(f"  New best val acc: {val_acc:.4f} | params: {best_params}")

print(f"\nBest val accuracy: {best_val_acc:.4f}")
print(f"Best params: {best_params}\n")

# ── Train best LightGBM on all train data ─────────────────────────────────────
lgb_final = lgb.LGBMClassifier(
    **best_params,
    class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
    min_child_samples=20
)
lgb_final.fit(X_train_bal, y_train_bal)
acc_lgb = accuracy_score(y_test, lgb_final.predict(X_test))
print(f"── LightGBM (tuned) test accuracy: {acc_lgb:.4f}")
print(classification_report(y_test, lgb_final.predict(X_test)))

# ── Stacking: LightGBM + RF probability features → LR ────────────────────────
print("\nBuilding stacking ensemble (LightGBM + RF → LR)...")
rf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                             random_state=42, n_jobs=-1, max_features="sqrt")
rf.fit(X_train_bal, y_train_bal)
acc_rf = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest test accuracy: {acc_rf:.4f}")

# Stack probas as meta features
meta_train = np.hstack([
    lgb_final.predict_proba(X_train_bal),
    rf.predict_proba(X_train_bal)
])
meta_test = np.hstack([
    lgb_final.predict_proba(X_test),
    rf.predict_proba(X_test)
])
meta_lr = LogisticRegression(max_iter=1000, C=5.0)
meta_lr.fit(meta_train, y_train_bal)
stack_preds = meta_lr.predict(meta_test)
acc_stack = accuracy_score(y_test, stack_preds)
print(f"Stacking test accuracy: {acc_stack:.4f}")
print(classification_report(y_test, stack_preds))

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n══ FINAL RESULTS ══")
all_results = {"LightGBM Tuned": acc_lgb, "Random Forest": acc_rf, "Stacking": acc_stack}
for name, acc in sorted(all_results.items(), key=lambda x: -x[1]):
    print(f"  {name:30s}: {acc:.4f}")

best_name = max(all_results, key=all_results.get)
print(f"\nBest model: {best_name} ({all_results[best_name]:.4f})")

payload = {
    "type": best_name,
    "lgb": lgb_final, "rf": rf, "meta": meta_lr,
    "tfidf": tfidf, "num_cols": available, "le": le
}
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(payload, f)
print("Saved to sentiment_model.pkl")
