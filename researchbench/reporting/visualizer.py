import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

def plot_target_distribution(target_dist: dict, target_name: str) -> str:
    if not target_dist or "counts" not in target_dist:
        return ""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = target_dist["counts"]
    ax.bar(list(map(str, counts.keys())), list(counts.values()), color='#4C72B0')
    ax.set_title(f"Target Distribution: {target_name}")
    ax.set_ylabel("Count")
    return _fig_to_base64(fig)

def plot_cv_boxplot(model_results: dict) -> str:
    # Build data for boxplot
    data = []
    labels = []
    for m, res in model_results.items():
        if "cv" in res and res["cv"] is not None:
            data.append(res["cv"]["folds"])
            labels.append(m)
    if not data:
        return ""
        
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, patch_artist=True)
    ax.set_xticklabels(labels)
    ax.set_title("Cross-Validation Fold Distribution")
    ax.set_ylabel("Metric Score")
    plt.xticks(rotation=45)
    return _fig_to_base64(fig)
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
