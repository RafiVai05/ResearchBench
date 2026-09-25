import pandas as pd
import numpy as np

def profile_dataset(df: pd.DataFrame, target: str = None) -> dict:
    """
    Profile the dataset.
    Returns rows, columns, numeric/categorical counts, missing values, duplicates, etc.
    """
    num_rows, num_cols = df.shape
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    missing_count = int(df.isnull().sum().sum())
    missing_percentage = (missing_count / (num_rows * num_cols)) * 100 if num_rows > 0 else 0
    
    duplicate_rows = int(df.duplicated().sum())
    
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    
    # Potential identifier columns (unique ratio = 1.0, or specific names)
    id_like_cols = []
    for col in df.columns:
        if df[col].nunique() == num_rows and num_rows > 1:
            id_like_cols.append(col)
        elif 'id' in str(col).lower() or 'uuid' in str(col).lower() or 'name' in str(col).lower():
            if col not in id_like_cols:
                id_like_cols.append(col)
                
    target_distribution = None
    if target and target in df.columns:
        counts = df[target].value_counts(dropna=False)
        percentages = (counts / num_rows) * 100
        target_distribution = {
            "counts": counts.to_dict(),
            "percentages": percentages.to_dict()
        }

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "missing_count": missing_count,
        "missing_percentage": missing_percentage,
        "duplicate_rows": duplicate_rows,
        "constant_cols": constant_cols,
        "id_like_cols": id_like_cols,
        "target_distribution": target_distribution,
        "target": target
    }