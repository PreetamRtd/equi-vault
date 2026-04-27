import pandas as pd
import numpy as np

def apply_k_anonymity(df, qi_columns, domain_rules, k=5):
    """
    Applies k-anonymity by generalizing QIs according to dynamic domain rules, 
    and suppressing groups that still don't meet the 'k' threshold.
    """
    df_anon = df.copy()
    
    # 1. Apply Dynamic Domain Rules (The "Blender Recipe")
    for col in qi_columns:
        if col in domain_rules and col in df_anon.columns:
            rule = domain_rules[col]
            
            # If the rule is numeric binning (e.g., Age or Income)
            if rule['type'] == 'binning':
                bins = rule['bins']
                labels = rule['labels']
                # Convert column to numeric, coercing errors to NaN
                df_anon[col] = pd.to_numeric(df_anon[col], errors='coerce')
                df_anon[col] = pd.cut(df_anon[col], bins=bins, labels=labels, right=False)
                
            # If the rule is date blurring (e.g., Date of Admission)
            elif rule['type'] == 'date_year_only':
                df_anon[col] = pd.to_datetime(df_anon[col], errors='coerce').dt.year.astype(str)
                
            # If the rule is direct suppression (e.g., Room Number or Zip Code)
            elif rule['type'] == 'suppress':
                df_anon[col] = 'Suppressed'

    # 2. Enforce k-anonymity via Suppression
    # Count the frequency of each QI combination
    qi_counts = df_anon.groupby(qi_columns).size().reset_index(name='count')
    # Keep only combinations that appear k or more times
    valid_groups = qi_counts[qi_counts['count'] >= k]
    # Merge back to drop the non-compliant rows
    df_k_anon = pd.merge(df_anon, valid_groups[qi_columns], on=qi_columns, how='inner')
    
    return df_k_anon

def apply_l_diversity(df, qi_columns, sensitive_column, domain_rules, k=5, l=2):
    """
    Applies l-diversity. First achieves k-anonymity, then filters out groups 
    that do not have at least 'l' distinct sensitive attributes.
    """
    # Step 1: Base it on k-anonymity
    df_k = apply_k_anonymity(df, qi_columns, domain_rules, k)
    
    # Step 2: Enforce l-diversity
    # Count unique sensitive attributes per QI group
    l_counts = df_k.groupby(qi_columns)[sensitive_column].nunique().reset_index(name='unique_sa')
    
    # Keep only groups meeting the 'l' threshold
    valid_l_groups = l_counts[l_counts['unique_sa'] >= l]
    df_l_diverse = pd.merge(df_k, valid_l_groups[qi_columns], on=qi_columns, how='inner')
    
    return df_l_diverse

def apply_t_closeness(df, qi_columns, sensitive_column, domain_rules, k=5, t=0.2):
    """
    MVP approximation of t-closeness. Ensures the distribution of the 
    sensitive attribute in a group does not deviate from the global distribution by more than 't'.
    """
    # Step 1: Base it on k-anonymity
    df_k = apply_k_anonymity(df, qi_columns, domain_rules, k)
    
    # Calculate global distribution of the sensitive attribute
    global_dist = df_k[sensitive_column].value_counts(normalize=True)
    
    valid_indices = []
    
    # Group by QIs and calculate local distributions
    grouped = df_k.groupby(qi_columns)
    for name, group in grouped:
        local_dist = group[sensitive_column].value_counts(normalize=True)
        
        # Calculate variation distance (simplified Earth Mover's Distance)
        max_deviation = 0
        for val in global_dist.index:
            local_prob = local_dist.get(val, 0)
            global_prob = global_dist[val]
            deviation = abs(local_prob - global_prob)
            if deviation > max_deviation:
                max_deviation = deviation
                
        # If the block is "close enough" to the global distribution, keep it
        if max_deviation <= t:
            valid_indices.extend(group.index)
            
    df_t_closed = df_k.loc[valid_indices]
    return df_t_closed

def apply_differential_privacy(df, numerical_sa_columns, epsilon=1.0):
    """
    Applies Differential Privacy by injecting Laplacian noise to numerical attributes.
    """
    df_dp = df.copy()
    
    for col in numerical_sa_columns:
        if col in df_dp.columns and pd.api.types.is_numeric_dtype(df_dp[col]):
            sensitivity = df_dp[col].max() - df_dp[col].min()
            if sensitivity == 0:
                continue # Skip if column has no variance
                
            scale = sensitivity / epsilon
            noise = np.random.laplace(loc=0.0, scale=scale, size=len(df_dp))
            df_dp[col] = np.clip(df_dp[col] + noise, 0, None)
            
    return df_dp