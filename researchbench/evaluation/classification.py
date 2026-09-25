from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
import numpy as np

def get_classification_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(random_state=42),
        "knn": KNeighborsClassifier()
    }

def evaluate_classification_metrics(y_true, y_pred, y_prob=None):
    classes = np.unique(y_true)
    is_multiclass = len(classes) > 2
    
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(y_true, y_pred, average='macro'),
        "Weighted F1": f1_score(y_true, y_pred, average='weighted'),
        "Macro Precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "Macro Recall": recall_score(y_true, y_pred, average='macro', zero_division=0)
    }
    
    if y_prob is not None:
        try:
            if is_multiclass:
                metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
            else:
                metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob)
        except Exception:
            metrics["ROC-AUC"] = None
    else:
        metrics["ROC-AUC"] = None
        
    cm = confusion_matrix(y_true, y_pred)
    
    return metrics, cm