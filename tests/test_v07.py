import pytest
import os
import json
import subprocess

def test_v07_end_to_end():
    config = {
        "dataset": "examples/classification.csv",
        "target": "target",
        "task": "classification",
        "models": {
            "logistic_regression": {},
            "random_forest": {}
        },
        "xai": {"enabled": True},
        "slice_finder": {"enabled": True},
        "ood_detection": {"enabled": True},
        "evaluation": {
            "strategy": "time_series",
            "time_column": "age"
        }
    }
    with open("test_conf_v07.json", "w") as out:
        json.dump(config, out)
        
    res = subprocess.run(["python", "-m", "researchbench.cli", "run", "--config", "test_conf_v07.json", "--save"], capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
    assert res.returncode == 0
    
    assert os.path.exists("researchbench-report.html")
    with open("researchbench-report.html", "r", encoding="utf-8") as f:
        html = f.read()
        assert "Global Surrogate Explainer" in html
        assert "Rule-Based Error Slicing" in html
        assert "Out-of-Distribution" in html