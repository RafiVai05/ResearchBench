import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc

def generate_confusion_matrix_plot(y_true, y_pred):
    """
    Generates a Plotly confusion matrix in JSON format.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                        labels=dict(x="Predicted Label", y="True Label", color="Count"))
        fig.update_layout(title="Confusion Matrix", width=600, height=500)
        return json.dumps(fig.to_plotly_json())
    except Exception:
        return None

def generate_roc_curve_plot(y_true, y_prob):
    """
    Generates a Plotly ROC curve in JSON format.
    """
    try:
        if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]
            
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(dash='dash')))
        fig.update_layout(title="Receiver Operating Characteristic", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", width=600, height=500)
        return json.dumps(fig.to_plotly_json())
    except Exception:
        return None