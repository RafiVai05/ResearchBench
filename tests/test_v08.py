import pytest
import os
import json
import subprocess
import pandas as pd

def test_v08_end_to_end():
    # create a dataset with text and schema violations
    df = pd.DataFrame({
        "age": [10, 20, 30, 40, 150, -5, 25, 35, 45, 55, 10, 20, 30, 40, 150, -5, 25, 35, 45, 55], # 150 and -5 are violations
        "notes": ["patient is doing fine today"] * 20, # unstructured text
        "target": [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
    })
    
    # make notes unique so it triggers auto_nlp
    df["notes"] = df["notes"] + " " + df["age"].astype(str)
    
    df.to_csv("examples/v08_data.csv", index=False)
    
    config = {
        "dataset": "examples/v08_data.csv",
        "target": "target",
        "task": "classification",
        "models": {
            "logistic_regression": {}
        },
        "conformal": {"enabled": True},
        "local_xai": {"enabled": True},
        "auto_nlp": {"enabled": True},
        "schema": {
            "age": {"min": 0, "max": 120}
        }
    }
    with open("test_conf_v08.json", "w") as out:
        json.dump(config, out)
        
    res = subprocess.run(["python", "-m", "researchbench.cli", "run", "--config", "test_conf_v08.json", "--save"], capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
    assert res.returncode == 0
    
    assert os.path.exists("researchbench-report.html")
    with open("researchbench-report.html", "r", encoding="utf-8") as f:
        html = f.read()
        assert "CRITICAL: Data Integrity Schema Violations" in html
        assert "rows above maximum bound (120)" in html
        assert "rows below minimum bound (0)" in html
        assert "Local Feature Attribution" in html