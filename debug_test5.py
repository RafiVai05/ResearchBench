from researchbench.audit.audit import perform_research_audit
import pandas as pd
df = pd.read_csv('examples/v108_data.csv')
audit_res = perform_research_audit(df, 'target', 'classification', {})
print("Type:", type(audit_res['health']['profile'].get('correlation_heatmap')))