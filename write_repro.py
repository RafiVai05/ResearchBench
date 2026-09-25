import os

with open('researchbench/audit/reproducibility.py', 'r', encoding='utf-8') as f:
    repro = f.read()

replacement = '''import sys
import platform
import datetime
from researchbench.dataset.fingerprint import get_environment_metadata

def get_reproducibility_info() -> dict:
    env = get_environment_metadata()
    env["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    return env
'''

with open('researchbench/audit/reproducibility.py', 'w', encoding='utf-8') as f:
    f.write(replacement)