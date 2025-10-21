"""Credit risk classifier (simple pipeline)"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

def build_pipeline(random_state=42):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=random_state))
    ])
    return pipe

def train_credit_model(X_train, y_train, X_val=None, y_val=None):
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    if X_val is not None and y_val is not None:
        preds = pipe.predict(X_val)
        probs = pipe.predict_proba(X_val)[:,1]
        print(classification_report(y_val, preds))
        print('ROC AUC:', roc_auc_score(y_val, probs))
    return pipe

def save_model(pipe, path):
    joblib.dump(pipe, path)

def load_model(path):
    return joblib.load(path)
