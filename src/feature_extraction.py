import librosa
import numpy as np
from transformers import pipeline

emotion_pipe = pipeline(task="audio-classification", model="aisak-ai/ED")

LABEL_MAP = {
    "LABEL_0": "sadness",
    "LABEL_1": "angry",
    "LABEL_2": "disgust",
    "LABEL_3": "fear",
    "LABEL_4": "happy",
    "LABEL_5": "neutral"
}

def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        rms = librosa.feature.rms(y=y).mean()
        pitch, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        mean_pitch = np.nanmean(pitch) if np.any(~np.isnan(pitch)) else 0.0
        return duration, mean_pitch, rms
    except Exception as e:
        print(f"[Audio feature error] {file_path}: {e}")
        return 0.0, 0.0, 0.0

def extract_emotion_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        predictions = emotion_pipe({"array": y, "sampling_rate": sr})
        # initialize all emotions with 0
        emotion_scores = {label: 0.0 for label in LABEL_MAP.values()}
        for p in predictions:
            label = LABEL_MAP.get(p["label"], p["label"])
            emotion_scores[label] = float(p["score"])
        return emotion_scores
    except Exception as e:
        print(f"[Emotion feature error] {file_path}: {e}")
        return {label: 0.0 for label in LABEL_MAP.values()}

def build_feature_vector(audio_file):
    duration, pitch, loudness = extract_audio_features(audio_file)
    emotion_scores = extract_emotion_features(audio_file)

    features = {"duration": duration, "pitch": pitch, "loudness": loudness}
    features.update(emotion_scores)
    return features
