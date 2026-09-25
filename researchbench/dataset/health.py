import pandas as pd
from .profiler import profile_dataset

def audit_dataset_health(df: pd.DataFrame, target: str = None) -> dict:
    """
    Perform a health audit on the dataset, finding potential concerns.
    """
    profile = profile_dataset(df, target)
    concerns = []
    observations = []
    
    n, p = profile["num_rows"], profile["num_cols"]
    
    if profile["missing_percentage"] > 0:
        concerns.append(f"{profile['missing_percentage']:.2f}% missing values detected.")
    
    if profile["duplicate_rows"] > 0:
        concerns.append(f"{profile['duplicate_rows']} duplicate rows detected.")
        
    if profile["constant_cols"]:
        concerns.append(f"Constant columns detected: {', '.join(profile['constant_cols'])}")
    else:
        observations.append("No constant columns.")
        
    if profile["id_like_cols"]:
        if target and target in profile["id_like_cols"]:
            pass # Target can have high cardinality if regression, handle elsewhere
        else:
            concerns.append(f"Potential identifier columns detected: {', '.join(profile['id_like_cols'])}")
            
    # Feature to sample ratio
    if n > 0 and (p / n) > 0.2:
        concerns.append(f"High feature-to-sample ratio ({p}/{n}). Risk of overfitting.")
        
    # Identical features (expensive for large df, do basic check if small)
    if p < 500:
        # Check for duplicated columns
        dup_cols = set()
        for i in range(p):
            col_i = df.columns[i]
            if col_i in dup_cols: continue
            for j in range(i+1, p):
                col_j = df.columns[j]
                if df[col_i].equals(df[col_j]):
                    dup_cols.add(col_j)
        if dup_cols:
            concerns.append(f"Duplicate columns detected: {', '.join(dup_cols)}")
            
    return {
        "profile": profile,
        "concerns": concerns,
        "observations": observations
    }