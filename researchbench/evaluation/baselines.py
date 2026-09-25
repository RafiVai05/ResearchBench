from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
import pandas as pd
import numpy as np

def evaluate_baseline_classification(X, y):
    clf = DummyClassifier(strategy='most_frequent')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Basic evaluate on whole dataset for metrics, but we also do CV elsewhere
    clf.fit(X, y)
    preds = clf.predict(X)
    return preds, clf

def get_baseline_models(task: str):
    if task == "classification":
        return {
            "Baseline (Most Frequent)": DummyClassifier(strategy='most_frequent'),
            "Baseline (Stratified)": DummyClassifier(strategy='stratified', random_state=42)
        }
    else:
        return {
            "Baseline (Mean)": DummyRegressor(strategy='mean'),
            "Baseline (Median)": DummyRegressor(strategy='median')
        }