import pandas as pd

def check_target_distribution(df: pd.DataFrame, target: str, task: str) -> dict:
    """
    Analyze target distribution, returning imbalance flags or outlier info.
    """
    if target not in df.columns:
        return {}
        
    y = df[target]
    result = {}
    
    if task == "classification":
        counts = y.value_counts()
        total = len(y)
        min_class = counts.min()
        min_class_ratio = min_class / total if total > 0 else 0
        
        result["is_imbalanced"] = min_class_ratio < 0.20
        result["is_severe_imbalance"] = min_class_ratio < 0.05
        result["min_class_ratio"] = min_class_ratio
        
    elif task == "regression":
        # Outlier detection using IQR
        q1 = y.quantile(0.25)
        q3 = y.quantile(0.75)
        iqr = q3 - q1
        outliers = ((y < (q1 - 1.5 * iqr)) | (y > (q3 + 1.5 * iqr))).sum()
        result["outlier_count"] = int(outliers)
        result["outlier_ratio"] = float(outliers / len(y)) if len(y) > 0 else 0
        result["has_outliers"] = result["outlier_ratio"] > 0.05
        
    return result