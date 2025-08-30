import os
import pandas as pd
from feature_extraction import build_feature_vector

data = []
folders = {"normal": "../data/normal", "fraud": "../data/fraud"} 

# Explicit feature names
feature_names = ["duration", "pitch", "loudness", "emotion_1", "emotion_2", "emotion_3"]

for label, folder in folders.items():
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            file_path = os.path.join(folder, file)
            features = build_feature_vector(file_path)
            row = {"file": file, "label": label}
            # assign features with proper names
            for name, val in zip(feature_names, features):
                row[name] = val
            data.append(row)

df = pd.DataFrame(data)
df.to_csv("multimodal_dataset.csv", index=False)
print(f"Dataset created: {len(df)} rows")
