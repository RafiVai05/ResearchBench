import os
import pandas as pd
import json
import subprocess

with open("researchbench-report.html", "r", encoding="utf-8") as f:
    content = f.read()
    if "Feature Correlation Matrix" in content:
        print("FOUND!")
    else:
        print("NOT FOUND!")