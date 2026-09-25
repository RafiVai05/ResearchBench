import pandas as pd
import os

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Safely load a dataset from a CSV file.
    Treats input as untrusted: no code execution, just standard parsing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {e}")