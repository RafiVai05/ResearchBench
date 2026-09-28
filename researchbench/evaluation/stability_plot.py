import json
import plotly.graph_objects as go

def generate_stability_plot(model_res):
    """
    Generates a Plotly box plot for cross-validation stability.
    """
    try:
        fig = go.Figure()
        
        for model_name, data in model_res.items():
            scores = data.get("cv", {}).get("scores", [])
            if scores:
                fig.add_trace(go.Box(x=scores, name=model_name, orientation='h'))
                
        if len(fig.data) == 0:
            return None
            
        fig.update_layout(title="Cross-Validation Stability", xaxis_title="Score", width=800, height=400)
        return json.dumps(fig.to_plotly_json())
    except Exception:
        return None