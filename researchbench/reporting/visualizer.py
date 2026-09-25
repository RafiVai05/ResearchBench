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
    std_resid = np.abs(np.array(residuals) - np.mean(residuals)) / (np.std(residuals) + 1e-9)
    sqrt_std_resid = np.sqrt(std_resid)
    ax4.scatter(preds, sqrt_std_resid, alpha=0.5, color="#27ae60")
    # Add trend line
    if len(preds) > 1:
        z = np.polyfit(preds, sqrt_std_resid, 1)
        p = np.poly1d(z)
        ax4.plot(preds, p(preds), color="red", linestyle="--")
    ax4.set_xlabel("Predicted Values")
    ax4.set_ylabel("Sqrt(Abs(Standardized Residuals))")
    ax4.set_title("Scale-Location")
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def plot_calibration_curve(calibration_results, model_name, output_dir):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        
        bins = calibration_results.get("bins", [])
        if not bins:
            return None
            
        prob_preds = [b["avg_prob"] for b in bins]
        prob_trues = [b["avg_acc"] for b in bins]
        
        plt.plot(prob_preds, prob_trues, marker='o', linewidth=2, label=model_name)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        
        ece = calibration_results.get("ece", 0)
        brier = calibration_results.get("brier_score", 0)
        plt.title(f"Reliability Diagram (ECE: {ece:.4f}, Brier: {brier:.4f})")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return _fig_to_base64(plt.gcf())
    except Exception as e:
        return None

def plot_permutation_importance(attribution_results, model_name, output_dir):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        features = attribution_results.get("features", [])
        means = attribution_results.get("importances_mean", [])
        stds = attribution_results.get("importances_std", [])
        
        if not features or not means:
            return None
            
        # Sort by mean
        indices = np.argsort(means)[::-1]
        sorted_features = [features[i] for i in indices]
        sorted_means = [means[i] for i in indices]
        sorted_stds = [stds[i] for i in indices]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sorted_means)), sorted_means, yerr=sorted_stds, align='center', alpha=0.8, ecolor='black', capsize=5)
        plt.xticks(range(len(sorted_means)), sorted_features, rotation=45, ha='right')
        plt.title(f"Permutation Feature Importance (CV Hold-out) - {model_name}")
        plt.ylabel("Mean Importance (Decrease in Metric)")
        plt.tight_layout()
        
        return _fig_to_base64(plt.gcf())
    except Exception:
        return None
