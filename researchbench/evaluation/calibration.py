import numpy as np

def calculate_calibration(y_true, y_prob, n_bins=10):
    if y_prob is None or len(y_prob) == 0:
        return None
        
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Handle multiclass by extracting max probability
    if y_prob.ndim == 2 and y_prob.shape[1] > 2:
        y_prob = np.max(y_prob, axis=1)
        # Brier score for multiclass is sum of squared differences across all classes
        # But for ECE and reliability diagrams, we often use the top-class probability
        pass
    elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]
    
    # Brier Score
    if y_prob.ndim == 1:
        # Binary
        brier = np.mean((y_prob - y_true)**2)
    else:
        # Multiclass full brier score
        # Needs one-hot encoding of y_true
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=np.arange(y_prob.shape[1]))
        if y_true_bin.shape[1] == 1: # Fallback
            y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
        brier = np.mean(np.sum((y_prob - y_true_bin)**2, axis=1))

    # ECE Calculation
    if y_prob.ndim > 1:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true)
    else:
        confidences = y_prob
        accuracies = y_true
        
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(confidences, bins, right=True)
    
    bin_stats = []
    ece = 0.0
    
    for b in range(1, len(bins)):
        bin_idx = (indices == b)
        if not np.any(bin_idx):
            continue
            
        bin_confs = confidences[bin_idx]
        bin_accs = accuracies[bin_idx]
        
        avg_conf = np.mean(bin_confs)
        avg_acc = np.mean(bin_accs)
        count = len(bin_confs)
        
        bin_stats.append({
            "bin_lower": bins[b-1],
            "bin_upper": bins[b],
            "count": count,
            "avg_prob": float(avg_conf),
            "avg_acc": float(avg_acc),
            "gap": float(np.abs(avg_conf - avg_acc))
        })
        
        ece += (count / len(confidences)) * np.abs(avg_conf - avg_acc)
        
    return {
        "ece": float(ece),
        "brier_score": float(brier),
        "n_bins": n_bins,
        "bins": bin_stats
    }