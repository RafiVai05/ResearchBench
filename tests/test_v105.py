import pytest
import os
import pandas as pd
import json
import subprocess

def test_v105_end_to_end():
    df = pd.DataFrame({
        "f1": list(range(10)),
        "target": [0, 1] * 5
    })
    df.to_csv("examples/v105_data.csv", index=False)
    
    config = {
        "dataset": "examples/v105_data.csv",
        "target": "target",
        "task": "classification",
        "models": {
            "logistic_regression": {}
        }
    }
    
    with open("test_conf_v105.json", "w") as out:
        json.dump(config, out)
        
    res = subprocess.run(["python", "-m", "researchbench.cli", "run", "--config", "test_conf_v105.json"], capture_output=True, text=True)
    assert res.returncode == 0
    assert os.path.exists("researchbench-report.md")
    
    with open("researchbench-report.md", "r") as f:
        content = f.read()
        assert "ResearchBench Report" in content
        assert "logistic_regression" in content