import pytest
import os
import pandas as pd
import numpy as np
from researchbench.api import ResearchBenchProject

def test_v102_103_end_to_end():
    # Dataset with strong redundancy (PCA) and outliers
    df = pd.DataFrame({
        "f1": list(range(100)) + [9999, -9999], # Add outliers
        "f2": list(range(100)) + [9999, -9999], # Perfect collinearity
        "target": [0]*51 + [1]*51  
    })
    
    config = {
        "dataset": "dataframe",
        "target": "target",
        "task": "classification",
        "models": {
            "logistic_regression": {}
        }
    }
    
    project = ResearchBenchProject(df, config)
    results = project.run()
    
    assert "audit" in results
    assert "outlier_fraction" in results["audit"]
    assert "pca_info" in results["audit"]
    assert results["audit"]["pca_info"]["n_components_95"] == 1
    assert results["audit"]["pca_info"]["total_numeric_features"] == 2