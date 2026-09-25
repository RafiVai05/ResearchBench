import os

with open('researchbench/reporting/visualizer.py', 'r', encoding='utf-8') as f:
    vis = f.read()

replacement = '''def generate_residual_plots(preds, residuals):
    if not preds or not residuals: return ""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    import base64
    from scipy import stats
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residuals vs Predicted
    ax1.scatter(preds, residuals, alpha=0.5, color="#2c3e50")
    ax1.axhline(0, color="red", linestyle="--")
    ax1.set_xlabel("Predicted Values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Predicted")
    
    # Residual Distribution
    ax2.hist(residuals, bins=30, color="#3498db", edgecolor="black")
    ax2.set_xlabel("Residual Value")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residual Distribution")
    
    # Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title("Q-Q Plot")
    
    # Scale-Location
    std_resid = np.abs(np.array(residuals) - np.mean(residuals)) / np.std(residuals)
    sqrt_std_resid = np.sqrt(std_resid)
    ax4.scatter(preds, sqrt_std_resid, alpha=0.5, color="#27ae60")
    # Add trend line
    z = np.polyfit(preds, sqrt_std_resid, 1)
    p = np.poly1d(z)
    ax4.plot(preds, p(preds), color="red", linestyle="--")
    ax4.set_xlabel("Predicted Values")
    ax4.set_ylabel("$\sqrt{|Standardized Residuals|}$")
    ax4.set_title("Scale-Location")
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
'''

import re
vis = re.sub(r'def generate_residual_plots.*?return base64\.b64encode.*?decode\("utf-8"\)', replacement, vis, flags=re.DOTALL)

with open('researchbench/reporting/visualizer.py', 'w', encoding='utf-8') as f:
    f.write(vis)