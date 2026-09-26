import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def check_dimensionality_pca(df, target=None):
    concerns = []
    pca_info = {"n_components_95": None, "total_numeric_features": 0}
    
    numeric_df = df.select_dtypes(include=[np.number])
    if target and target in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target])
        
    # Drop zero variance columns
    numeric_df = numeric_df.loc[:, numeric_df.var() > 0]
    
    if numeric_df.empty or numeric_df.shape[1] < 2:
        return concerns, pca_info
        
    try:
        clean_df = numeric_df.fillna(numeric_df.median())
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(clean_df)
        
        pca = PCA()
        pca.fit(scaled_df)
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_comp = np.argmax(cumulative_variance >= 0.95) + 1
        
        pca_info["n_components_95"] = int(n_comp)
        pca_info["total_numeric_features"] = numeric_df.shape[1]
        
        if n_comp < numeric_df.shape[1] * 0.1 and numeric_df.shape[1] > 10:
            concerns.append(f"High redundancy: 95% of variance in {numeric_df.shape[1]} features can be explained by just {n_comp} PCA components.")
            
    except Exception as e:
        pass
        
    return concerns, pca_info