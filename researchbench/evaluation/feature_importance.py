import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def generate_feature_importance_plot(pipeline, feature_names):
    try:
        # Extract estimator
        if hasattr(pipeline, "_final_estimator"):
            estimator = pipeline._final_estimator
        elif hasattr(pipeline, "steps"):
            estimator = pipeline.steps[-1][1]
        else:
            estimator = pipeline
            
        importances = None
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            importances = np.abs(estimator.coef_[0]) if estimator.coef_.ndim > 1 else np.abs(estimator.coef_)
            
        if importances is None or len(importances) == 0:
            return None
            
        # Match dimensions (in case of one-hot encoding, feature_names length might not match)
        if len(importances) != len(feature_names):
            return None
            
        # Get top 10 features
        indices = np.argsort(importances)[::-1][:10]
        top_importances = importances[indices]
        top_features = [feature_names[i] for i in indices]
        
        plt.figure(figsize=(6, 4))
        plt.title("Top 10 Feature Importances")
        plt.barh(range(len(top_features)), top_importances[::-1], align="center", color="steelblue")
        plt.yticks(range(len(top_features)), top_features[::-1])
        plt.xlabel("Importance / Absolute Coefficient")
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        return None