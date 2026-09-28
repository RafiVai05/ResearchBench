from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, matthews_corrcoef, brier_score_loss, average_precision_score
import numpy as np
import warnings

def get_classification_models():
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42)
    }
    
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
    except ImportError:
        pass
        
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(random_state=42)
    except ImportError:
        pass
        
    try:
        from catboost import CatBoostClassifier
        models["catboost"] = CatBoostClassifier(random_state=42, verbose=0)
    except ImportError:
        pass
        
    return models

def evaluate_classification_metrics(y_true, y_pred, y_prob=None, config=None):
    metrics = {}
    
    # Accuracy
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        
    metrics["Macro Precision"] = p_macro
    metrics["Macro Recall"] = r_macro
    metrics["Macro F1"] = f1_macro
    metrics["Weighted F1"] = f1_weighted
    
    # Matthews Correlation Coefficient (MCC) - V1.1.0
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred)
    
    # Balanced Accuracy
    from sklearn.metrics import balanced_accuracy_score
    metrics["Balanced Accuracy"] = balanced_accuracy_score(y_true, y_pred)
    
    # Probability Metrics
    if y_prob is not None:
        try:
            # ROC AUC
            if y_prob.shape[1] == 2:
                metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob[:, 1])
                metrics["PR-AUC"] = average_precision_score(y_true, y_prob[:, 1])
                # Brier Score (only for binary directly)
                metrics["Brier Score"] = brier_score_loss(y_true, y_prob[:, 1])
            else:
                metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except Exception:
            metrics["ROC-AUC"] = 0.0
            metrics["PR-AUC"] = 0.0
            
    cm = confusion_matrix(y_true, y_pred)
    return metrics, cm