import os
import re

with open('researchbench/advisor/advisor.py', 'r', encoding='utf-8') as f:
    adv = f.read()

replacement = '''
    report = {
        "observations": [],
        "potential_concerns": [],
        "evidence": [],
        "suggested_investigations": [],
        "reproducibility_notes": [],
        "limitations": []
    }
    
    # 1. Leakage Analysis
    if "concerns" in leakage and leakage["concerns"]:
        for c in leakage["concerns"]:
            if "TEMPORAL LEAKAGE RISK" in c or "GROUP LEAKAGE RISK" in c:
                report["potential_concerns"].append(c)
                report["suggested_investigations"].append("Re-evaluate validation strategy (e.g. use TimeSeriesSplit or GroupKFold).")
            else:
                report["potential_concerns"].append(c)
                report["suggested_investigations"].append("Check dataset for overlapping samples.")
'''

adv = re.sub(r'    report = \{\n        "observations": \[\],\n.*?"limitations": \[\]\n    \}\n    \n    # 1. Leakage Analysis\n    if "concerns" in leakage and leakage\["concerns"\]:\n        for c in leakage\["concerns"\]:\n            report\["potential_concerns"\].append\(c\)\n            report\["suggested_investigations"\].append\("Check dataset construction for duplicate rows"\)', replacement.strip(), adv, flags=re.DOTALL)

with open('researchbench/advisor/advisor.py', 'w', encoding='utf-8') as f:
    f.write(adv)