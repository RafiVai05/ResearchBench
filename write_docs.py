import os

with open('CHANGELOG.md', 'a', encoding='utf-8') as f:
    f.write('''
## [0.2.0] - 2026-09-25

### Added
- External YAML Configuration Support
- Automatic Preprocessing Pipelines (ColumnTransformer inside CV)
- Hyperparameter tuning support via GridSearchCV 
- Regression Residual Analysis module
- Local Experiment History Vault (--save, history command)
- Change-Aware Research Advisor (compare command)
- New HTML Report Visualizations for Hyperparameters and Preprocessing

### Changed
- Refactored core audit logic to accept configurable thresholds from YAML
- Updated CLI to accept --preprocess, --config, and --save arguments
- Cross-validation and stability loops now index Pandas DataFrames natively
''')

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

# Update version in readme
readme = readme.replace('RESEARCHBENCH v0.1', 'RESEARCHBENCH v0.2')

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme)