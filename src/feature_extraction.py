import librosa
import numpy as np
from transformers import pipeline

# Load emotion model
emotion_pipe = pipeline(model="aisak-ai/ED")
LABEL_MAP = {
    "LABEL_0": "sadness",
    "LABEL_1": "angry",
    "LABEL_2": "disgust",
    "LABEL_3": "fear",
    "LABEL_4": "happy",
    "LABEL_5": "neutral"
}

def best_emotional_mix(emotions, top_n=3):
    sorted_emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)
    total_score = sum(e['score'] for e in sorted_emotions)
    for e in sorted_emotions[:top_n]:
        e['percentage'] = (e['score'] / total_score) * 100 if total_score > 0 else 0
    return sorted_emotions[:top_n]

def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        rms = librosa.feature.rms(y=y).mean()
        pitch, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
        mean_pitch = np.nanmean(pitch)
        return duration, mean_pitch, rms
    except:
        return 0.0, 0.0, 0.0

def extract_emotion_features(file_path, top_n=3):
    try:
        predictions = emotion_pipe(file_path)
        mapped = [{"label": LABEL_MAP[p["label"]], "score": p["score"]} for p in predictions]
        top_emotions = best_emotional_mix(mapped, top_n=top_n)
        return [e['score'] for e in top_emotions]
    except:
        return [0.0] * top_n

def build_feature_vector(audio_file):
    duration, pitch, loudness = extract_audio_features(audio_file)
    emotion_scores = extract_emotion_features(audio_file, top_n=3)
    return np.array([duration, pitch, loudness] + emotion_scores, dtype=float)
