def check_sample_size(num_rows: int, num_cols: int) -> list:
    concerns = []
    if num_rows < 50:
        concerns.append("Very small sample size (< 50). Results may not generalize.")
    if num_rows > 0 and (num_cols / num_rows) > 0.5:
        concerns.append(f"Very high feature-to-sample ratio ({num_cols}/{num_rows}). High risk of overfitting without regularization.")
    return concerns