
import pytest
import pandas as pd
import numpy as np

def test_calibration():
    from researchbench.evaluation.calibration import calculate_calibration
    y_true = [0, 1, 0, 1, 0]
    y_prob = [0.1, 0.9, 0.2, 0.8, 0.3]
    res = calculate_calibration(y_true, y_prob, n_bins=5)
    assert res is not None
    assert "ece" in res
    assert "brier_score" in res
    assert res["brier_score"] < 0.2
    
def test_drift():
    from researchbench.audit.drift import detect_drift
    df = pd.DataFrame({"num": np.random.randn(100), "cat": np.random.choice(["A", "B"], 100)})
    df.to_csv("test_ext.csv", index=False)
    
    config = {"drift_detection": {"enabled": True, "external_dataset": "test_ext.csv"}}
    df_train = pd.DataFrame({"num": np.random.randn(100) + 2.0, "cat": np.random.choice(["C", "D"], 100)})
    
    res = detect_drift(df_train, config)
    assert res is not None
    
    import os
    os.remove("test_ext.csv")

def test_sensitivity():
    from researchbench.evaluation.sensitivity import run_sensitivity_analysis
    from sklearn.linear_model import LogisticRegression
    
    df = pd.DataFrame({"f1": np.random.randn(50), "f2": np.random.randn(50)})
    y = np.random.randint(0, 2, 50)
    
    model = LogisticRegression()
    config = {
        "evaluation": {"cv": {"folds": 2}},
        "sensitivity_analysis": {
            "enabled": True,
            "parameters": {"C": [0.1, 1.0, 10.0]}
        }
    }
    
    res = run_sensitivity_analysis(model, df, y, "classification", config)
    assert res is not None

def test_export_and_artifacts():
    import json
    import subprocess
    config = {
        "dataset": "examples/classification.csv",
        "target": "target",
        "task": "classification",
        "models": {"logistic_regression": {}},
        "artifact_audit": {"enabled": True}
    }
    with open("test_conf_v05.json", "w") as out:
        json.dump(config, out)
        
    res = subprocess.run(["python", "-m", "researchbench.cli", "run", "--config", "test_conf_v05.json", "--save"], capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
    assert res.returncode == 0
    
    res_export = subprocess.run(["python", "-m", "researchbench.cli", "export", ".researchbench/history.json", "--outdir", "test_exports"], capture_output=True, text=True)
    assert res_export.returncode == 0
    
    import shutil
    import os
    shutil.rmtree("test_exports", ignore_errors=True)
    os.remove("test_conf_v05.json")
