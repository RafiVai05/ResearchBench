import os
import subprocess
import json
import pandas as pd
import pytest

def test_v109_end_to_end():
    df = pd.DataFrame({
        "f1": list(range(20)),
        "f2": [0]*20,
        "target": [0, 1] * 10
    })
    os.makedirs("examples", exist_ok=True)
    df.to_csv("examples/v109_data.csv", index=False)
    
    config = {
        "dataset": "examples/v109_data.csv",
        "target": "target",
        "task": "classification",
        "feature_selection": {
            "enabled": True,
            "k": 1,
            "strategy": "mutual_info"
        },
        "optimization": {
            "enabled": True,
            "n_trials": 2
        },
        "models": {
            "logistic_regression": {}
        }
    }
    
    with open("test_conf_v109.json", "w") as out:
        json.dump(config, out)
        
    res = subprocess.run(["python", "-m", "researchbench.cli", "run", "--config", "test_conf_v109.json"], capture_output=True, text=True)
    assert res.returncode == 0
    
    with open("researchbench-report.html", "r", encoding="utf-8") as f:
        content = f.read()
        assert "tabcontent" in content
        assert "tablinks" in content