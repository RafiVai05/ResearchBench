import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

def generate_confusion_matrix_plot(y_true, y_pred):
    try:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=ax, colorbar=False)
        plt.title("Confusion Matrix (OOF)")
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        return None

def generate_roc_curve_plot(y_true, y_probs):
    try:
        # Check if binary classification
        unique_classes = np.unique(y_true)
        if len(unique_classes) != 2:
            return None
            
        # Ensure y_probs is 1D (probability of positive class)
        y_probs = np.array(y_probs)
        if y_probs.ndim > 1:
            if y_probs.shape[1] == 2:
                y_probs = y_probs[:, 1]
            else:
                return None
                
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (OOF)')
        ax.legend(loc="lower right")
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        return None