import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# load dataset
df = pd.read_csv("multimodal_dataset.csv")
X = df[[c for c in df.columns if c.startswith("feature_")]].values
y = df["label"].map({"normal": 0, "fraud": 1}).values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# training
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# save
joblib.dump(clf, "../models/multimodal_model.pkl")
print("Model saved as models/multimodal_model.pkl")

# eval
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1] * 100  # probability % of fraud

print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Sample fraud probabilities (%):", y_prob[:5])
