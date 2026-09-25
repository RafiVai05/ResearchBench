import os
import re

with open('pyproject.toml', 'r', encoding='utf-8') as f:
    pyproj = f.read()

pyproj = pyproj.replace('version = "0.2.0"', 'version = "0.3.0"')
pyproj += '''
[project.optional-dependencies]
stats = [
    "statsmodels>=0.14.0"
]
'''

with open('pyproject.toml', 'w', encoding='utf-8') as f:
    f.write(pyproj)

with open('researchbench/__init__.py', 'r', encoding='utf-8') as f:
    init = f.read()

init = init.replace("__version__ = '0.2.0'", "__version__ = '0.3.0'")
with open('researchbench/__init__.py', 'w', encoding='utf-8') as f:
    f.write(init)