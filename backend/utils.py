import numpy as np
import json
import os

def clean_json_types(obj):
    """
    Recursively converts Numpy data types to standard Python types.
    Prevents FastAPI from crashing with 'Object of type int64 is not JSON serializable'.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: clean_json_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_types(i) for i in obj]
    return obj

def load_domain_rules(domain: str):
    """Loads the specific JSON generalization hierarchies based on the chosen domain."""
    filename = f"{domain.lower()}_rules.json"
    filepath = os.path.join("rules", filename)
    
    # Fallback mock rules for local testing if the file isn't created yet
    if not os.path.exists(filepath):
        if domain.lower() == "healthcare":
            return {
                "Age": {"type": "binning", "bins": [0,20,40,60,80,120], "labels": ["0-20","21-40","41-60","61-80","80+"]},
                "Date of Admission": {"type": "date_year_only"},
                "Discharge Date": {"type": "date_year_only"},
                "Hospital": {"type": "suppress"},
                "Room Number": {"type": "suppress"}
            }
        return {}
        
    with open(filepath, 'r') as f:
        return json.load(f)