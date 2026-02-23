import ssl
import os
import requests
import pandas as pd
import whisper

# Fix SSL cert verification on macOS Python 3.10
ssl._create_default_https_context = ssl._create_unverified_context

# Load Whisper model (use "base" for speed, "small" or "medium" for better accuracy)
model = whisper.load_model("base")

df = pd.read_csv("raw_f1_dataset.csv")

os.makedirs("audio_files", exist_ok=True)

# Resume from where we left off if transcript column already exists
if "transcript" not in df.columns:
    df["transcript"] = None

transcripts = df["transcript"].tolist()

for i, row in df.iterrows():
    # Skip already transcribed
    if pd.notna(transcripts[i]):
        continue

    url = row.get("recording_url")

    if pd.isna(url) or not isinstance(url, str):
        continue

    filename = os.path.join("audio_files", f"{row['session_key']}_{row['driver_number']}_{i}.mp3")

    # Download audio if not already cached
    if not os.path.exists(filename):
        try:
            r = requests.get(url, timeout=20, verify=False)
            if r.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(r.content)
            else:
                continue
        except Exception as e:
            print(f"  Download error row {i}: {e}")
            continue

    # Transcribe
    try:
        result = model.transcribe(filename)
        transcript = result["text"].strip()
        transcripts[i] = transcript
        print(f"[{i}/{len(df)}] {row.get('name_acronym', row['driver_number'])}: {transcript}")
    except Exception as e:
        print(f"  Transcription error row {i}: {e}")

    # Save progress every 50 rows
    if i % 50 == 0:
        df["transcript"] = transcripts
        df.to_csv("raw_f1_dataset.csv", index=False)

df["transcript"] = transcripts
df.to_csv("raw_f1_dataset.csv", index=False)
print(f"\nDone. {df['transcript'].notna().sum()} / {len(df)} transcribed.")
