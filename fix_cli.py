import os

# Fix formatting emojis
with open('researchbench/utils/formatting.py', 'r', encoding='utf-8') as f:
    fmt = f.read()
fmt = fmt.replace('⚠', '[WARNING]').replace('ℹ', '[INFO]').replace('✓', '[OK]')
with open('researchbench/utils/formatting.py', 'w', encoding='utf-8') as f:
    f.write(fmt)

# Fix CLI model evaluation dropping non-numeric
with open('researchbench/cli.py', 'r', encoding='utf-8') as f:
    cli = f.read()

replacement = \"\"\"
            y = df[args.target]
            X = df.drop(columns=[args.target])
            # For v0.1 drop non-numeric
            import numpy as np
            X = X.select_dtypes(include=[np.number])
\"\"\"

cli = cli.replace(\"\"\"
            y = df[args.target]
            X = df.drop(columns=[args.target])
\"\"\", replacement)

with open('researchbench/cli.py', 'w', encoding='utf-8') as f:
    f.write(cli)

print("Fixed CLI and Formatting")