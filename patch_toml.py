import re

with open('pyproject.toml', 'r', encoding='utf-8') as f:
    text = f.read()

# Add optional dependencies
opt_deps = '''
[project.optional-dependencies]
boosting = ["xgboost", "lightgbm", "catboost"]
plotly = ["plotly"]
conformal = ["mapie"]
artifacts = ["PyMuPDF"]
'''

text += opt_deps

# Update version
text = re.sub(r'version = "1\.0\.\d+"', 'version = "1.1.0"', text)

with open('pyproject.toml', 'w', encoding='utf-8') as f:
    f.write(text)

with open('researchbench/__init__.py', 'r', encoding='utf-8') as f:
    text = f.read()
text = re.sub(r'__version__ = "1\.0\.\d+"', '__version__ = "1.1.0"', text)
with open('researchbench/__init__.py', 'w', encoding='utf-8') as f:
    f.write(text)