import re

with open('pyproject.toml', 'r', encoding='utf-8') as f:
    text = f.read()
text = re.sub(r'version = "1\.0\.\d+"', 'version = "1.0.8"', text)
with open('pyproject.toml', 'w', encoding='utf-8') as f:
    f.write(text)

with open('researchbench/__init__.py', 'r', encoding='utf-8') as f:
    text = f.read()
text = re.sub(r'__version__ = "1\.0\.\d+"', '__version__ = "1.0.8"', text)
with open('researchbench/__init__.py', 'w', encoding='utf-8') as f:
    f.write(text)