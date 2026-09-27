import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def generate_stability_plot(model_results):
    try:
        model_names = []
        model_folds = []
        
        for name, data in model_results.items():
            if "cv" in data and "folds" in data["cv"]:
                model_names.append(name)
                model_folds.append(data["cv"]["folds"])
                
        if len(model_names) == 0:
            return None
            
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(model_folds, vert=False, patch_artist=True, 
                   boxprops=dict(facecolor='lightblue', color='blue'),
                   medianprops=dict(color='red'))
        
        ax.set_yticks(np.arange(1, len(model_names) + 1))
        ax.set_yticklabels(model_names)
        
        ax.set_title('Cross-Validation Score Stability')
        ax.set_xlabel('Score')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        return None