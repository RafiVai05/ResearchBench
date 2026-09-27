import pytest
import os
import pandas as pd
import json
import subprocess

def test_v108_end_to_end():
    df = pd.DataFrame({
        "f1": list(range(20)),
        "target": [0, 1] * 10
    })
    df.to_csv("examples/v108_data.csv", index=False)
    
    config = {
        "dataset": "examples/v108_data.csv",
        "target": "target",
        "task": "classification",
        "models": {
            "logistic_regression": {},
            "random_forest": {}
        }
    }
    
    with open("test_conf_v108.json", "w") as out:
        json.dump(config, out)
        
    res = subprocess.run(["python", "-m", "researchbench.cli", "run", "--config", "test_conf_v108.json"], capture_output=True, text=True)
    assert res.returncode == 0
    
    with open("researchbench-report.html", "r", encoding="utf-8") as f:
        content = f.read()
        assert "Cross-Validation Stability (All Models)" in content