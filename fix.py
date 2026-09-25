import os
for root, dirs, files in os.walk('researchbench'):
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            if r'\"\"\"' in content:
                content = content.replace(r'\"\"\"', '\"\"\"')
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f'Fixed {path}')