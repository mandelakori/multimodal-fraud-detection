import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# load dataset
df = pd.read_csv("multimodal_dataset.csv")
X = df[["duration", "pitch", "loudness", "emotion_1", "emotion_2", "emotion_3"]].values
y = df["label"].map({"normal": 0, "fraud": 1}).values

# handle class imbalance by weighting samples according to duration
sample_weights = df["duration"].values
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# interpolate weights to match resampled dataset
resampled_weights = np.interp(np.linspace(0, len(sample_weights)-1, len(y_res)), np.arange(len(sample_weights)), sample_weights)

# Train/test split
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_res, y_res, resampled_weights, test_size=0.2, random_state=42, stratify=y_res
)

# Random Forest hyperparameter grid
param_dist = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False]
}

# Randomized search for best hyperparameters
rf_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(class_weight="balanced", random_state=42),
    param_distributions=param_dist,
    n_iter=20,  # number of random combinations
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# training with duration weights
rf_search.fit(X_train, y_train, sample_weight=w_train)

# best estimator
clf = rf_search.best_estimator_
print("Best parameters:", rf_search.best_params_)
print(f"Best CV ROC-AUC: {rf_search.best_score_:.3f}")

# save
joblib.dump(clf, "../models/multimodal_model.pkl")
print("Model saved as models/multimodal_model.pkl")

# eval
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1] * 100  # probability % of fraud

print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Sample fraud probabilities (%):", y_prob[:5])

# cross-validation for ROC-AUC on full resampled dataset
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_res, y_res, cv=cv, scoring="roc_auc")
print(f"5-Fold CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
