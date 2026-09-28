from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def get_regression_models():
    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(random_state=42),
        "decision_tree": DecisionTreeRegressor(random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = XGBRegressor(random_state=42)
    except ImportError:
        pass
        
    try:
        from lightgbm import LGBMRegressor
        models["lightgbm"] = LGBMRegressor(random_state=42)
    except ImportError:
        pass
        
    try:
        from catboost import CatBoostRegressor
        models["catboost"] = CatBoostRegressor(random_state=42, verbose=0)
    except ImportError:
        pass
        
    return models

def evaluate_regression_metrics(y_true, y_pred, config=None):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }