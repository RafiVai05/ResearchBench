import pytest
import os
import json
import pandas as pd
from researchbench.api import ResearchBenchProject

def test_v10_end_to_end():
    df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 2,
        "f2": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 2,
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 2
    })
    
    config = {
        "dataset": "dataframe",
        "target": "target",
        "task": "classification",
        "models": {
            "logistic_regression": {},
            "random_forest": {}
        },
        "ensembling": {"enabled": True}
    }
    
    # Test programmatic API
    project = ResearchBenchProject(df, config)
    results = project.run()
    
    assert "audit" in results
    assert "models" in results
    
    # Test dataset hashing
    assert "dataset_hash" in results["audit"]["reproducibility"]
    
    # Test ensembling
    assert "ensemble_voting" in results["models"]
    
    # Test CLI for joblib serialization
    df.to_csv("examples/v10_data.csv", index=False)
    config["dataset"] = "examples/v10_data.csv"
    with open("test_conf_v10.json", "w") as out:
        json.dump(config, out)
        
    import subprocess
    res = subprocess.run(["python", "-m", "researchbench.cli", "run", "--config", "test_conf_v10.json", "--save"], capture_output=True, text=True)
    assert res.returncode == 0
    
    assert os.path.exists("researchbench-report.html")
    assert os.path.exists("researchbench-report_best_model.joblib")
    
    # Verify joblib loading
    import joblib
    model = joblib.load("researchbench-report_best_model.joblib")
    assert hasattr(model, "predict")