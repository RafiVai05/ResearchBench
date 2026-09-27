import os
import pandas as pd
import json
import subprocess

df = pd.DataFrame({
    "f1": list(range(100)),
    "target": [0, 1] * 50
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
print("OUT:", res.stdout[:500])
print("ERR:", res.stderr)

with open("researchbench-report.html", "r", encoding="utf-8") as f:
    content = f.read()
    if "Cross-Validation Stability" in content:
        print("FOUND!")
    else:
        print("NOT FOUND!")