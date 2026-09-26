import pytest
import os
import pandas as pd
from researchbench.api import ResearchBenchProject

def test_v101_end_to_end():
    # Highly imbalanced dataset to trigger auto-balancing
    df = pd.DataFrame({
        "f1": list(range(100)),
        "f2": list(range(100)),
        "target": [0]*90 + [1]*10  # 90% vs 10%
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
    assert "models" in results
    
    # Check if learning curve was generated
    assert "logistic_regression" in results["models"]
    assert "learning_curve" in results["models"]["logistic_regression"]
    assert results["models"]["logistic_regression"]["learning_curve"].startswith("data:image/png;base64,")