import pandas as pd
import numpy as np

np.random.seed(42)
n = 100
df = pd.DataFrame({
    'Age': np.random.randint(20, 70, n),
    'Gender': np.random.choice(['M', 'F'], n),
    'Patient_ID': np.random.randint(1, 10, n),  # Multiple records per patient
    'Review': np.random.choice(['Good response to treatment', 'Bad reaction', 'No change', 'Excellent recovery'], n),
    'target': np.random.randint(0, 2, n)
})
df.to_csv('dummy2.csv', index=False)