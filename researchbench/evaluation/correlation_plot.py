import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import warnings

def generate_correlation_heatmap(df):
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None
            
        # Limit to top 15 features to prevent massive unreadable plots
        if numeric_df.shape[1] > 15:
            # We just take the first 15 for visualization
            numeric_df = numeric_df.iloc[:, :15]
            
        corr = numeric_df.corr()
        
        # Mask upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Custom simple heatmap using matplotlib (no seaborn dependency)
        cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='left')
        ax.set_yticklabels(corr.columns)
        
        # Loop over data dimensions and create text annotations.
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                if not mask[i, j]:
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                            ha="center", va="center", color="black", fontsize=8)
                            
        ax.xaxis.set_ticks_position('bottom')
        plt.title('Feature Correlation Heatmap', pad=20)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        return None