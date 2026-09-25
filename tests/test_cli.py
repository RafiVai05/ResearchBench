import subprocess
import os

def test_cli_profile():
    res = subprocess.run(["python", "-m", "researchbench.cli", "profile", "examples/classification.csv", "--target", "target"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Rows: 1000" in res.stdout