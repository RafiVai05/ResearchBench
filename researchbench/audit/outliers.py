import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_outliers_isolation_forest(df, target=None):
    concerns = []
    outlier_fraction = 0.0
    
    # Use only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if target and target in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target])
        
    if numeric_df.empty or numeric_df.shape[1] == 0:
        return concerns, outlier_fraction
        
    try:
        # Fill NaNs with median for isolation forest
        clean_df = numeric_df.fillna(numeric_df.median())
        iso_forest = IsolationForest(contamination='auto', random_state=42)
        preds = iso_forest.fit_predict(clean_df)
        
        # -1 means outlier, 1 means inlier
        num_outliers = np.sum(preds == -1)
        outlier_fraction = num_outliers / len(preds)
        
        if outlier_fraction > 0.05:
            concerns.append(f"High anomaly rate detected: {outlier_fraction*100:.1f}% of data points flagged as multivariate outliers by Isolation Forest.")
            
    except Exception as e:
        pass
        
    return concerns, outlier_fraction