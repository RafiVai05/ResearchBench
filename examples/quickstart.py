import pandas as pd
import numpy as np

# Generate Synthetic Classification Data
np.random.seed(42)
n_samples = 1000

# Create realistic clinical diagnostic features
age = np.random.normal(55, 10, n_samples)
biomarker1 = np.random.normal(10, 2, n_samples)
biomarker2 = np.random.normal(5, 1.5, n_samples)
id_col = np.array([f"patient_{i}" for i in range(n_samples)])

# Target logic with some noise
target_prob = 1 / (1 + np.exp(-(-5 + 0.1 * age + 0.5 * biomarker1 - 0.2 * biomarker2)))
diagnosis = (target_prob > 0.5).astype(int)
# introduce imbalance
diagnosis[:800] = 0

df_class = pd.DataFrame({
    'patient_id': id_col,
    'age': age,
    'biomarker1': biomarker1,
    'biomarker2': biomarker2,
    'target': diagnosis
})
# Inject some missing values
df_class.loc[10:30, 'biomarker1'] = np.nan

df_class.to_csv("examples/classification.csv", index=False)

# Regression
price = 50000 + 1000 * age + 500 * biomarker1
# Add outliers
price[5:15] = price[5:15] * 3

df_reg = pd.DataFrame({
    'id': id_col,
    'age': age,
    'biomarker1': biomarker1,
    'target': price
})
df_reg.to_csv("examples/regression.csv", index=False)