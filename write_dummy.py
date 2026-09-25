import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'Age': np.random.randint(20, 70, 100),
    'Income': np.random.uniform(20000, 100000, 100),
    'Feature3': np.random.randn(100),
    'target': np.random.randint(0, 2, 100)
})
df.to_csv('dummy.csv', index=False)