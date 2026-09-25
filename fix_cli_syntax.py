import os
import re

with open('researchbench/cli.py', 'r', encoding='utf-8') as f:
    cli = f.read()

cli = cli.replace('print(\"\\nPossible interpretation:\")', 'print(\"\\\\nPossible interpretation:\")')
cli = cli.replace('print(\"\\nSuggested investigation:\")', 'print(\"\\\\nSuggested investigation:\")')
cli = cli.replace('print(f\"Rows: {prof[\\'num_rows\\']}\\\\nColumns: {prof[\\'num_cols\\']}\\\\n\")', 'print(f\"Rows: {prof[\\'num_rows\\']}\\nColumns: {prof[\\'num_cols\\']}\\n\")')

# I will just write a small cleaner script to fix syntax errors
try:
    compile(cli, 'cli.py', 'exec')
except SyntaxError as e:
    print(f"Syntax error still exists! {e}")
    # I'll manually fix the lines
    
with open('researchbench/cli.py', 'w', encoding='utf-8') as f:
    f.write(cli)