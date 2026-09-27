import pytest
import os
import pandas as pd
from researchbench.api import ResearchBenchProject

def test_v107_end_to_end():
    # Binary classification dataset to trigger CM and ROC
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
            "logistic_regression": {}
        }
    }
    
    project = ResearchBenchProject(df, config)
    results = project.run()
    
    assert "models" in results
    assert "logistic_regression" in results["models"]
    
    m_res = results["models"]["logistic_regression"]
    
    assert "confusion_matrix" in m_res
    assert m_res["confusion_matrix"].startswith("data:image/png;base64,")
    
    assert "roc_curve" in m_res
    assert m_res["roc_curve"].startswith("data:image/png;base64,")