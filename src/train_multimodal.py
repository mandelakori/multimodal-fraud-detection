import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# load dataset
df = pd.read_csv("multimodal_dataset.csv")
X = df[[c for c in df.columns if c.startswith("feature_")]].values
y = df["label"].map({"normal": 0, "fraud": 1}).values

# handle class imbalance by weighting samples according to duration
sample_weights = df["duration"].values
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
resampled_weights = np.interp(np.arange(len(y_res)), np.arange(len(sample_weights)), sample_weights)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# training
clf = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)
clf.fit(X_train, y_train, sample_weight=resampled_weights)

# save
joblib.dump(clf, "../models/multimodal_model.pkl")
print("Model saved as models/multimodal_model.pkl")

# eval
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1] * 100  # probability % of fraud

print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Sample fraud probabilities (%):", y_prob[:5])

# cross-validation for ROC-AUC
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_res, y_res, cv=cv, scoring="roc_auc")
print(f"5-Fold CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
