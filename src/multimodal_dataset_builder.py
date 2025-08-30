import os
import pandas as pd
from feature_extraction import build_feature_vector

data = []
folders = {"normal": "data/normal", "fraud": "data/fraud"}

for label, folder in folders.items():
    if not os.path.exists(folder):
        print(f"⚠️ Warning: folder {folder} does not exist, skipping.")
        continue
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            file_path = os.path.join(folder, file)
            try:
                features = build_feature_vector(file_path)
                row = {"file": file, "label": label}
                row.update(features)
                data.append(row)
            except Exception as e:
                print(f"[Error] {file}: {e}")


df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
df.to_csv("multimodal_dataset.csv", index=False)
print(f"Dataset created: {len(df)} rows -> multimodal_dataset.csv")
