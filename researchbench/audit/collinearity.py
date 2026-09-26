import numpy as np
import pandas as pd

def check_collinearity(df: pd.DataFrame, threshold: float = 10.0) -> list:
    """
    Checks for high multicollinearity using Variance Inflation Factor (VIF) approximation.
    Uses inverse of correlation matrix to approximate VIFs.
    """
    concerns = []
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    
    if numeric_df.shape[1] < 2 or numeric_df.shape[0] < 10:
        return concerns
        
    try:
        # Calculate correlation matrix
        corr = numeric_df.corr().values
        
        # Calculate inverse of correlation matrix
        # Diagonal elements of inverse corr matrix are the VIFs
        inv_corr = np.linalg.inv(corr)
        vifs = np.diag(inv_corr)
        
        for idx, col in enumerate(numeric_df.columns):
            vif = vifs[idx]
            if vif > threshold:
                concerns.append(f"[COLLINEARITY] Feature '{col}' has high Variance Inflation Factor (VIF = {vif:.2f}). This destabilizes feature importance.")
                
    except np.linalg.LinAlgError:
        concerns.append("[COLLINEARITY] Correlation matrix is singular. Extreme multicollinearity exists (perfectly correlated features).")
    except Exception:
        pass
        
    return concerns
