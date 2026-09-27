from researchbench.dataset.loader import load_dataset
from researchbench.dataset.health import audit_dataset_health
import pandas as pd
df = pd.read_csv('examples/v108_data.csv')
audit = audit_dataset_health(df, 'target')
print('Heatmap in profile:', bool(audit['profile'].get('correlation_heatmap')))