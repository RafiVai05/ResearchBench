import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def mine_hard_examples(y_true, y_pred, y_prob, X, task, top_p=0.05):
    """
    Mines the top_p% hardest examples and clusters them to find error typologies.
    """
    if len(y_true) < 20:
        return None
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if task == "classification":
        if y_prob is not None:
            if y_prob.ndim == 2:
                # get prob of true class
                prob_true = y_prob[np.arange(len(y_true)), y_true]
                errors = 1.0 - prob_true
            else:
                prob_true = np.where(y_true == 1, y_prob, 1.0 - y_prob)
                errors = 1.0 - prob_true
        else:
            errors = (y_true != y_pred).astype(float)
    else:
        errors = np.abs(y_true - y_pred)
        
    n_top = max(1, int(len(errors) * top_p))
    top_indices = np.argsort(errors)[-n_top:][::-1]
    
    hard_samples = []
    if isinstance(X, pd.DataFrame):
        hard_df = X.iloc[top_indices].copy()
        hard_df["True_Label"] = y_true[top_indices]
        hard_df["Predicted"] = y_pred[top_indices]
        hard_df["Error_Magnitude"] = errors[top_indices]
        hard_samples = hard_df.to_dict(orient="records")
    else:
        # Numpy array fallback
        for i in top_indices:
            hard_samples.append({
                "index": int(i),
                "True_Label": float(y_true[i]),
                "Predicted": float(y_pred[i]),
                "Error_Magnitude": float(errors[i])
            })
            
    # Cluster the errors
    typology = []
    if isinstance(X, pd.DataFrame):
        try:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0 and len(top_indices) >= 3:
                X_hard = X.iloc[top_indices][numeric_cols].fillna(0)
                scaler = StandardScaler()
                X_hard_scaled = scaler.fit_transform(X_hard)
                
                n_clusters = min(3, len(top_indices))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(X_hard_scaled)
                
                centers = scaler.inverse_transform(kmeans.cluster_centers_)
                
                for c in range(n_clusters):
                    profile = {}
                    for idx, col in enumerate(numeric_cols):
                        profile[col] = float(centers[c, idx])
                    size = int(np.sum(kmeans.labels_ == c))
                    typology.append({
                        "cluster_id": c,
                        "size": size,
                        "centroid": profile
                    })
        except Exception:
            pass
            
    return {
        "top_k_count": n_top,
        "hard_examples": hard_samples[:10], # Top 10 for HTML
        "error_clusters": typology
    }
