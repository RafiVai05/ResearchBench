import os
import re

with open('pyproject.toml', 'r', encoding='utf-8') as f:
    pyproj = f.read()

pyproj = pyproj.replace('version = "0.3.0"', 'version = "0.4.0"')

if 'torch =' not in pyproj:
    pyproj = pyproj.replace(
        '[project.optional-dependencies]',
        '[project.optional-dependencies]\ntorch = [\n    "torch>=2.0.0"\n]\nhuggingface = [\n    "transformers>=4.0.0",\n    "datasets>=2.0.0"\n]'
    )

with open('pyproject.toml', 'w', encoding='utf-8') as f:
    f.write(pyproj)

with open('researchbench/__init__.py', 'r', encoding='utf-8') as f:
    init = f.read()

init = init.replace("__version__ = '0.3.0'", "__version__ = '0.4.0'")
with open('researchbench/__init__.py', 'w', encoding='utf-8') as f:
    f.write(init)