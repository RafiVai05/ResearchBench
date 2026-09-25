import os
import re

with open('pyproject.toml', 'r', encoding='utf-8') as f:
    pyproj = f.read()

# Just remove the last block we appended
pyproj = re.sub(r'\[project\.optional-dependencies\]\nstats = \[\n    "statsmodels>=0\.14\.0"\n\]\n', '', pyproj)

# Find if there's already an optional-dependencies block
if '[project.optional-dependencies]' in pyproj:
    pyproj = pyproj.replace('[project.optional-dependencies]', '[project.optional-dependencies]\nstats = ["statsmodels>=0.14.0"]')
else:
    pyproj += '''
[project.optional-dependencies]
stats = ["statsmodels>=0.14.0"]
'''

with open('pyproject.toml', 'w', encoding='utf-8') as f:
    f.write(pyproj)