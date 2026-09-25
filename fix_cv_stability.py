import os
import re

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
        
    code = code.replace("X_arr = np.array(X)", "X_arr = X")
    code = code.replace("y_arr = np.array(y)", "y_arr = y")
    
    # Replace indexing
    code = code.replace("X_arr[train_idx]", "X_arr.iloc[train_idx] if hasattr(X_arr, 'iloc') else X_arr[train_idx]")
    code = code.replace("X_arr[test_idx]", "X_arr.iloc[test_idx] if hasattr(X_arr, 'iloc') else X_arr[test_idx]")
    
    code = code.replace("y_arr[train_idx]", "y_arr.iloc[train_idx] if hasattr(y_arr, 'iloc') else y_arr[train_idx]")
    code = code.replace("y_arr[test_idx]", "y_arr.iloc[test_idx] if hasattr(y_arr, 'iloc') else y_arr[test_idx]")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(code)

fix_file('researchbench/evaluation/cross_validation.py')
fix_file('researchbench/evaluation/stability.py')