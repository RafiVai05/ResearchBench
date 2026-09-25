import os

with open('researchbench/cli.py', 'r', encoding='utf-8') as f:
    cli = f.read()

replacement = '''
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)

def save_experiment(record, filename=None):
    os.makedirs(".researchbench/experiments", exist_ok=True)
    if not filename:
        record["id"] = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        filename = f".researchbench/experiments/{record['id']}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=4, cls=NpEncoder)
'''
cli = cli.replace('''def save_experiment(record, filename=None):
    os.makedirs(".researchbench/experiments", exist_ok=True)
    if not filename:
        record["id"] = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        filename = f".researchbench/experiments/{record['id']}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=4)''', replacement)

with open('researchbench/cli.py', 'w', encoding='utf-8') as f:
    f.write(cli)