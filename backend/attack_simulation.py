import pandas as pd

def simulate_homogeneity_attack(df_anon, qi_columns, sensitive_column):
    """
    Calculates Re-identification Risk.
    If everyone in a generalized group shares the exact same sensitive attribute,
    an attacker knows the attribute of anyone in that group (100% exposed).
    """
    # Group by Quasi-Identifiers
    grouped = df_anon.groupby(qi_columns)
    
    exposed_rows = 0
    total_groups = 0
    vulnerable_groups = 0
    
    for name, group in grouped:
        total_groups += 1
        # If there is only 1 unique sensitive attribute in this group
        if group[sensitive_column].nunique() == 1:
            vulnerable_groups += 1
            exposed_rows += len(group)
            
    risk_percentage = (exposed_rows / len(df_anon)) * 100 if len(df_anon) > 0 else 0
    
    return {
        "exposed_records": exposed_rows,
        "vulnerable_classes": vulnerable_groups,
        "total_classes": total_groups,
        "risk_percentage": round(risk_percentage, 2),
        "warning": f"{exposed_rows} records are 100% exposed to Homogeneity Attacks." if exposed_rows > 0 else "Secure from Homogeneity."
    }

def simulate_skewness_attack(df_anon, qi_columns, sensitive_column, skew_threshold=0.90):
    """
    Calculates Skewness Risk (l-diversity weakness).
    If one attribute makes up >90% of a group, an attacker can guess it with 90% accuracy.
    """
    grouped = df_anon.groupby(qi_columns)
    
    highly_skewed_rows = 0
    
    for name, group in grouped:
        # Calculate the frequency of the most common sensitive attribute
        most_frequent_count = group[sensitive_column].value_counts().iloc[0]
        skew_ratio = most_frequent_count / len(group)
        
        if skew_ratio >= skew_threshold:
            highly_skewed_rows += len(group)
            
    risk_percentage = (highly_skewed_rows / len(df_anon)) * 100 if len(df_anon) > 0 else 0
    
    return {
        "highly_skewed_records": highly_skewed_rows,
        "skew_risk_percentage": round(risk_percentage, 2),
        "warning": f"{highly_skewed_rows} records sit in highly skewed classes (>90% probability)." if highly_skewed_rows > 0 else "Secure from Skewness."
    }