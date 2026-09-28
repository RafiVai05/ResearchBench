import numpy as np
import json
import plotly.express as px

def generate_correlation_heatmap(df):
    """
    Generates a Plotly heatmap for feature correlation.
    """
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None
            
        if numeric_df.shape[1] > 15:
            numeric_df = numeric_df.iloc[:, :15]
            
        corr = numeric_df.corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig.update_layout(title="Feature Correlation Matrix", width=700, height=600)
        return json.dumps(fig.to_plotly_json())
    except Exception:
        return None