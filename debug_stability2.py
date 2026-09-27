import numpy as np
import matplotlib.pyplot as plt
from researchbench.evaluation.stability_plot import generate_stability_plot
model_res = {
    "model_a": {"cv": {"folds": [0.8, 0.85, 0.82]}},
    "model_b": {"cv": {"folds": [0.9, 0.91, 0.89]}}
}
import traceback
try:
    generate_stability_plot(model_res)
except Exception as e:
    traceback.print_exc()

import os
with open('researchbench/evaluation/stability_plot.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('except Exception as e:\n        return None', 'except Exception as e:\n        import traceback\n        traceback.print_exc()\n        return None')
with open('researchbench/evaluation/stability_plot.py', 'w', encoding='utf-8') as f:
    f.write(text)