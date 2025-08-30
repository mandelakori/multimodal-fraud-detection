import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, average_precision_score
from imblearn.over_sampling import SMOTE
import joblib

# load dataset
df = pd.read_csv("multimodal_dataset.csv")
X = df[["duration", "pitch", "loudness", "emotion_1", "emotion_2", "emotion_3"]].values
y = df["label"].map({"normal": 0, "fraud": 1}).values

# handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# define base models
rf = RandomForestClassifier(
    n_estimators=400,
    min_samples_split=5,
    min_samples_leaf=2,
    max_depth=10,
    class_weight={0:1, 1:3},  # penalize fraud more
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    scale_pos_weight=3,  # heavier weight for fraud
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

lgbm = lgb.LGBMClassifier(
    n_estimators=400,
    max_depth=10,
    class_weight={0:1, 1:3},
    random_state=42
)

# stacking ensemble
stack = StackingClassifier(
    estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
    final_estimator=RandomForestClassifier(
        n_estimators=200,
        class_weight={0:1,1:3},
        random_state=42
    ),
    cv=5,
    passthrough=True
)

# train
stack.fit(X_train, y_train)

# save
joblib.dump(stack, "../models/multimodal_model.pkl")
print("Model saved as models/multimodal_model.pkl")

# eval
y_pred = stack.predict(X_test)
y_prob = stack.predict_proba(X_test)[:, 1] * 100  # probability % of fraud

print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Sample fraud probabilities (%):", y_prob[:5])

# additional fraud-focused metrics
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("Precision-Recall AUC:", average_precision_score(y_test, y_prob/100))

# cross-validation for ROC-AUC
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(stack, X_res, y_res, cv=cv, scoring="roc_auc")
print(f"5-Fold CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
