import json

def export_json(report_data: dict, output_path: str):
    """
    Export report data as structured machine-readable JSON.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4)