import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import learning_curve

def generate_learning_curve_plot(estimator, X, y, task, cv_folds=5):
    try:
        scoring = "accuracy" if task == "classification" else "r2"
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv_folds, n_jobs=-1, 
            train_sizes=np.linspace(0.2, 1.0, 5), scoring=scoring
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(6, 4))
        plt.title("Learning Curve")
        plt.xlabel("Training examples")
        plt.ylabel("Score (" + scoring + ")")
        plt.grid(True)
        
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        return None