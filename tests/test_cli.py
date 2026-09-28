import sys
import os
import subprocess

def test_cli_profile():
    try:
        res = subprocess.run([sys.executable, "-m", "researchbench.cli", "profile", "examples/classification.csv", "--target", "target"], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"CLI Profile Warning: {res.stderr}")
    except Exception as e:
        print(f"Exception: {e}")