import hashlib
import platform
import sys
import pandas as pd
import sklearn
import numpy as np

def generate_dataset_fingerprint(df: pd.DataFrame, target: str = None) -> dict:
    """Generate a reproducible metadata fingerprint of a dataset without storing raw data."""
    
    # We use a stable hash of the column names and shape
    cols = sorted(df.columns.tolist())
    cols_str = ",".join(cols)
    
    shape_str = f"{df.shape[0]}x{df.shape[1]}"
    
    # Optional: hash of a sample of data, but we stick to metadata for speed and safety
    # We can hash the data types
    dtypes_str = ",".join([str(df[c].dtype) for c in cols])
    
    raw_sig = f"{shape_str}|{cols_str}|{dtypes_str}"
    fingerprint_hash = hashlib.sha256(raw_sig.encode('utf-8')).hexdigest()[:16]
    
    return {
        "hash": fingerprint_hash,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "target": target
    }

def get_environment_metadata() -> dict:
    import researchbench
    
    metadata = {
        "researchbench_version": researchbench.__version__,
        "python_version": sys.version.split(" ")[0],
        "os": platform.system() + " " + platform.release(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__
    }
    
    try:
        import scipy
        metadata["scipy_version"] = scipy.__version__
    except ImportError:
        pass
        
    try:
        import statsmodels
        metadata["statsmodels_version"] = statsmodels.__version__
    except ImportError:
        pass
        
    return metadata