from abc import ABC, abstractmethod

class ModelWrapper(ABC):
    """
    Base class for wrapping models (e.g., PyTorch, external APIs, etc.)
    so they can be evaluated transparently by ResearchBench.
    """
    
    @abstractmethod
    def fit(self, X, y):
        pass
        
    @abstractmethod
    def predict(self, X):
        pass
        
    def predict_proba(self, X):
        raise NotImplementedError("This model does not support predict_proba")

class SklearnWrapper(ModelWrapper):
    def __init__(self, estimator):
        self.estimator = estimator
        
    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self
        
    def predict(self, X):
        return self.estimator.predict(X)
        
    def predict_proba(self, X):
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        return super().predict_proba(X)