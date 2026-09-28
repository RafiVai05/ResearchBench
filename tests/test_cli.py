import sys
import subprocess
import os

def test_cli_profile():
    res = subprocess.run([sys.executable, "-m", "researchbench.cli", "profile", "examples/classification.csv", "--target", "target"], capture_output=True, text=True)
    assert res.returncode == 0, f"CLI Failed!\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "Rows: 1000" in res.stdout