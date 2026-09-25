def check_metrics_suitability(task: str, has_imbalance: bool, has_outliers: bool) -> list:
    concerns = []
    if task == "classification" and has_imbalance:
        concerns.append("Accuracy may not fully represent minority-class performance. Consider Macro F1, Balanced Accuracy, and per-class recall.")
    if task == "regression" and has_outliers:
        concerns.append("MAE may provide a more robust view of typical prediction error than MSE alone due to outliers.")
    return concerns