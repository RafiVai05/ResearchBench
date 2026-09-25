import os

with open('researchbench/reporting/visualizer.py', 'r', encoding='utf-8') as f:
    content = f.read()
    
new_content = '''
def generate_residual_plots(preds, residuals):
    if not preds or not residuals: return ""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io
    import base64
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(preds, residuals, alpha=0.5, color="#2c3e50")
    ax1.axhline(0, color="red", linestyle="--")
    ax1.set_xlabel("Predicted Values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Predicted")
    
    ax2.hist(residuals, bins=30, color="#3498db", edgecolor="black")
    ax2.set_xlabel("Residual Value")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residual Distribution")
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
'''

if 'generate_residual_plots' not in content:
    with open('researchbench/reporting/visualizer.py', 'a', encoding='utf-8') as f:
        f.write(new_content)