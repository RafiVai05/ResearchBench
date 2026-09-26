import pytest
import os
import json
import subprocess
import pandas as pd

def test_v09_end_to_end():
    df = pd.DataFrame({
        "age": [10, 20, 30, 40, 50, 60, 70, 80, 25, 35, 45, 55, 65, 75, 85, 15, 22, 33, 44, 55],
        "age_dup": [10, 20, 30, 40, 50, 60, 70, 80, 25, 35, 45, 55, 65, 75, 85, 15, 22, 33, 44, 55], # triggers collinearity
        "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "F"], # protected attr
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] # females always 1, disparate impact ~ infinity
    })
    
    df.to_csv("examples/v09_data.csv", index=False)
    
    config = {
        "dataset": "examples/v09_data.csv",
        "target": "target",
        "task": "classification",
        "models": {
            "logistic_regression": {}
        },
        "fairness": {
            "protected_attributes": ["gender"],
            "privileged_classes": {"gender": "M"}
        },
        "optimization": {
            "enabled": True,
            "n_trials": 2
        }
    }
    with open("test_conf_v09.json", "w") as out:
        json.dump(config, out)
        
    res = subprocess.run(["python", "-m", "researchbench.cli", "run", "--config", "test_conf_v09.json", "--save"], capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
    assert res.returncode == 0
    
    assert os.path.exists("researchbench-report.html")
    with open("researchbench-report.html", "r", encoding="utf-8") as f:
        html = f.read()
        assert "Algorithmic Fairness" in html
        assert "Executive Insights" in html
        assert "Multicollinearity was detected" in html or "Correlation matrix is singular" in html
        assert "Disparate Impact" in html