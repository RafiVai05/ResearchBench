import sys
import os
import subprocess
import json
import pandas as pd
import pytest

def test_v110_end_to_end():
    df = pd.DataFrame({
        "f1": list(range(100)),
        "f2": [0]*100,
        "f3": ["A", "B", "C", "D"] * 25,
        "target": [0, 1] * 50
    })
    os.makedirs("examples", exist_ok=True)
    df.to_csv("examples/v110_data.csv", index=False)
    
    config = {
        "dataset": "examples/v110_data.csv",
        "target": "target",
        "task": "classification",
        "feature_selection": {
            "enabled": True,
            "k": 2,
            "strategy": "mutual_info"
        },
        "optimization": {
            "enabled": True,
            "n_trials": 2
        },
        "calibration": {
            "enabled": True
        },
        "artifact_audit": {
            "enabled": True
        },
        "models": {
            "logistic_regression": {},
            "random_forest": {}
        }
    }
    
    with open("test_conf_v110.json", "w") as out:
        json.dump(config, out)
        
    res = subprocess.run([sys.executable, "-m", "researchbench.cli", "run", "--config", "test_conf_v110.json"], capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
        print(res.stdout)
    assert res.returncode == 0
    
    with open("researchbench-report.html", "r", encoding="utf-8") as f:
        content = f.read()
        assert "plotly-2.27.0.min.js" in content
        assert "Research Artifact Audit (v1.1.0)" in content