import pytest
import os
import json
import subprocess

def test_v06_end_to_end():
    config = {
        "dataset": "examples/classification.csv",
        "target": "target",
        "task": "classification",
        "models": {
            "logistic_regression": {},
            "random_forest": {}
        },
        "compute_profiling": {"enabled": True},
        "robustness": {"enabled": True},
        "hard_examples": {"enabled": True},
        "agreement": {"enabled": True}
    }
    with open("test_conf_v06.json", "w") as out:
        json.dump(config, out)
        
    res = subprocess.run(["python", "-m", "researchbench.cli", "run", "--config", "test_conf_v06.json", "--save"], capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
    assert res.returncode == 0
    
    assert os.path.exists("researchbench-report.html")
    with open("researchbench-report.html", "r", encoding="utf-8") as f:
        html = f.read()
        assert "Computational Profiling &amp; Efficiency" in html or "Computational Profiling" in html
        assert "Empirical Robustness" in html
        assert "Hard Example Mining" in html
        assert "Model Agreement" in html