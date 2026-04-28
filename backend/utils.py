import numpy as np
import json
import os
import pandas as pd

def clean_json_types(obj):
    """Prevents FastAPI from crashing with Numpy JSON errors."""
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

def get_dynamic_rules(df, qi_cols):
    """
    Dynamically generates masking rules for ANY dataset.
    No hardcoded JSON files needed anymore!
    """
    rules = {}
    for col in qi_cols:
        if col not in df.columns:
            continue
            
        # If the column is numbers (like Age or Income), automatically bin it into 5 groups
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            
            # Create mathematical bins
            bins = np.linspace(min_val, max_val, 6).tolist()
            bins[0] = -float('inf') # Catch outliers below
            bins[-1] = float('inf') # Catch outliers above
            
            labels = [f"Group {i+1}" for i in range(5)]
            
            rules[col] = {
                "type": "binning",
                "bins": bins,
                "labels": labels
            }
        else:
            # If the column is text (like Occupation or Zip Code), suppress it
            rules[col] = {"type": "suppress"}
            
    return rules