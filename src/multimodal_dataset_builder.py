import os
import pandas as pd
from feature_extraction import build_feature_vector

data = []
folders = {"normal": "data/normal", "fraud": "data/fraud"} 
for label, folder in folders.items():
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            file_path = os.path.join(folder, file)
            features = build_feature_vector(file_path)
            row = {"file": file, "label": label}
            for i, val in enumerate(features, 1):
                row[f"feature_{i}"] = val
            data.append(row)

df = pd.DataFrame(data)
df.to_csv("multimodal_dataset.csv", index=False)
print(f"Dataset created: {len(df)} rows")
