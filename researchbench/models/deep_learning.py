from .base import ModelWrapper
import numpy as np
import logging

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class PyTorchWrapper(ModelWrapper):
    def __init__(self, model, config=None, task="classification"):
        if not HAS_TORCH:
            raise ImportError("torch is required for PyTorchWrapper. Please install researchbench[torch]")
            
        self.model = model
        self.task = task
        self.config = config.get("deep_learning", {}) if config else {}
        self.epochs = self.config.get("epochs", 10)
        self.batch_size = self.config.get("batch_size", 32)
        self.lr = self.config.get("learning_rate", 0.001)
        
        device_str = self.config.get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
            
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        if self.task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
            
    def _prepare_data(self, X, y=None):
        X_arr = X.values if hasattr(X, "values") else np.array(X)
        if y is not None:
            y_arr = y.values if hasattr(y, "values") else np.array(y)
            return torch.tensor(X_arr, dtype=torch.float32), torch.tensor(y_arr, dtype=torch.long if self.task == "classification" else torch.float32)
        return torch.tensor(X_arr, dtype=torch.float32)

    def fit(self, X, y):
        X_t, y_t = self._prepare_data(X, y)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                
                if self.task == "classification":
                    loss = self.criterion(outputs, batch_y)
                else:
                    loss = self.criterion(outputs.squeeze(), batch_y)
                    
                loss.backward()
                self.optimizer.step()
                
        return self

    def predict(self, X):
        X_t = self._prepare_data(X)
        loader = DataLoader(TensorDataset(X_t), batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        
        preds = []
        with torch.no_grad():
            for batch_x in loader:
                outputs = self.model(batch_x[0].to(self.device))
                if self.task == "classification":
                    _, predicted = torch.max(outputs, 1)
                    preds.append(predicted.cpu().numpy())
                else:
                    preds.append(outputs.squeeze().cpu().numpy())
                    
        return np.concatenate(preds)

    def predict_proba(self, X):
        if self.task != "classification":
            raise NotImplementedError("predict_proba is for classification tasks")
            
        X_t = self._prepare_data(X)
        loader = DataLoader(TensorDataset(X_t), batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        
        probs = []
        with torch.no_grad():
            for batch_x in loader:
                outputs = self.model(batch_x[0].to(self.device))
                probabilities = torch.softmax(outputs, dim=1)
                probs.append(probabilities.cpu().numpy())
                
        return np.concatenate(probs)

try:
    from transformers import Pipeline as HFPipeline
    HAS_HF = True
except ImportError:
    HAS_HF = False

class HuggingFaceWrapper(ModelWrapper):
    def __init__(self, pipeline, config=None):
        if not HAS_HF:
            raise ImportError("transformers is required for HuggingFaceWrapper. Please install researchbench[huggingface]")
        self.pipeline = pipeline
        self.config = config.get("deep_learning", {}) if config else {}
        
    def fit(self, X, y):
        # Hugging Face wrappers here typically expect pre-trained pipelines for inference evaluation,
        # or require standard Trainer configurations. We assume evaluation mode for this wrapper format.
        logging.warning("HuggingFaceWrapper fit() called, assuming pre-trained model.")
        return self
        
    def predict(self, X):
        X_texts = X.iloc[:, 0].tolist() if hasattr(X, "iloc") else list(X)
        results = self.pipeline(X_texts)
        # Extract labels from pipeline format depending on task
        # Simplified:
        if isinstance(results[0], dict) and "label" in results[0]:
            return np.array([int(r["label"].split("_")[-1]) if r["label"].startswith("LABEL_") else r["label"] for r in results])
        return np.array(results)