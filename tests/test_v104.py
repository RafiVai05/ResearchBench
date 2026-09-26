import pytest
import os
import pandas as pd
from researchbench.api import ResearchBenchProject

def test_v104_end_to_end():
    # Basic numeric dataset
    df = pd.DataFrame({
        "f1": list(range(100)),
        "f2": list(range(100)),
        "target": [0]*50 + [1]*50
    })
    
    config = {
        "dataset": "dataframe",
        "target": "target",
        "task": "classification",
        "models": {
            "logistic_regression": {},
            "random_forest": {}
        }
    }
    
    project = ResearchBenchProject(df, config)
    results = project.run()
    
    assert "models" in results
    assert "logistic_regression" in results["models"]
    assert "feature_importance" in results["models"]["logistic_regression"]
    assert results["models"]["logistic_regression"]["feature_importance"].startswith("data:image/png;base64,")
    
    assert "random_forest" in results["models"]
    assert "feature_importance" in results["models"]["random_forest"]
    assert results["models"]["random_forest"]["feature_importance"].startswith("data:image/png;base64,")