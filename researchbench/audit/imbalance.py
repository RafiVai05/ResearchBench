import pandas as pd

def check_imbalance(df: pd.DataFrame, target: str, task: str, config: dict = None) -> list:
    concerns = []
    if task != "classification" or target not in df.columns:
        return concerns
        
    counts = df[target].value_counts()
    total = len(df)
    if total == 0: return concerns
    
    min_class_ratio = counts.min() / total
    
    if min_class_ratio < 0.05:
        concerns.append("Severe class imbalance detected (< 5%). Accuracy will be misleading.")
    elif min_class_ratio < 0.20:
        concerns.append("Class imbalance detected (< 20%). Consider whether accuracy adequately represents performance.")
        
    return concerns