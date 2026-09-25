import pandas as pd

def audit_data_schema(df: pd.DataFrame, schema_config: dict) -> list:
    """
    Checks if the dataset satisfies predefined constraints.
    """
    concerns = []
    if not schema_config:
        return concerns
        
    for col, constraints in schema_config.items():
        if col not in df.columns:
            concerns.append(f"[SCHEMA] Missing required column: '{col}'")
            continue
            
        series = df[col].dropna()
        
        if "min" in constraints:
            if (series < constraints["min"]).any():
                count = (series < constraints["min"]).sum()
                concerns.append(f"[SCHEMA] Column '{col}' has {count} rows below minimum bound ({constraints['min']}).")
                
        if "max" in constraints:
            if (series > constraints["max"]).any():
                count = (series > constraints["max"]).sum()
                concerns.append(f"[SCHEMA] Column '{col}' has {count} rows above maximum bound ({constraints['max']}).")
                
        if "allowed_values" in constraints:
            invalid = ~series.isin(constraints["allowed_values"])
            if invalid.any():
                count = invalid.sum()
                concerns.append(f"[SCHEMA] Column '{col}' has {count} rows with invalid categorical values.")
                
    return concerns
