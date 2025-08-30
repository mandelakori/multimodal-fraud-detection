import os
import pandas as pd
import joblib
from src.feature_extraction import build_feature_vector

def main():
    print("=== Multimodal Fraud Detection ===")
    audio_path = input("Enter path to audio file: ").strip()

    if not os.path.isfile(audio_path):
        print(f"Error: File '{audio_path}' does not exist.")
        return

    # load stacked model
    try:
        model = joblib.load("models/multimodal_model.pkl")
    except Exception as e:
        print("Error loading model:", e)
        return

    # extract features
    try:
        features = build_feature_vector(audio_path)
        print("Extracted features:", features)  # debug output
        feature_names = ["duration", "pitch", "loudness", "sadness", "angry", "disgust", "fear", "happy", "neutral"]
        features_df = pd.DataFrame([features], columns=feature_names)
        print("Features DataFrame:\n", features_df)  # debug output
    except Exception as e:
        print("Error extracting features:", e)  
        return

    # prediction
    try:
        prob_fraud = model.predict_proba(features_df)[0][1] * 100  # fraud probability %
        prob_normal = 100 - prob_fraud
    except Exception as e:
        print("Error during prediction:", e)
        return

    print(f"Fraud probability: {prob_fraud:.2f}%")
    print(f"Normal probability: {prob_normal:.2f}%")

if __name__ == "__main__":
    main()
