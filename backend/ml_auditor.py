import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from fairlearn.metrics import demographic_parity_difference

# Import ALL techniques from the upgraded privacy engine
from privacy_engine import apply_k_anonymity, apply_l_diversity, apply_t_closeness, apply_differential_privacy

def preprocess_for_ml(df, target_col):
    """Encodes text/categorical variables into numbers for the ML model."""
    df_encoded = df.copy()
    
    # Instantly drop direct identifiers (expandable list)
    identifiers = ['Name', 'name', 'Patient ID', 'SSN', 'id']
    for id_col in identifiers:
        if id_col in df_encoded.columns:
            df_encoded = df_encoded.drop(columns=[id_col])
    
    # BULLETPROOF ENCODING: Check if the column is NOT a number
    for col in df_encoded.columns:
        if not pd.api.types.is_numeric_dtype(df_encoded[col]):
            le = LabelEncoder()
            # Convert to string first to prevent mixed-type errors, then encode to integers
            df_encoded[col] = df_encoded[col].astype(str)
            df_encoded[col] = le.fit_transform(df_encoded[col])
            
    return df_encoded

def run_audit(df, target_col, protected_col, qi_cols, num_sa_cols, domain_rules):
    """Runs the complete AI Proving Ground pipeline with dynamic domain rules."""
    
    # 1. SPLIT FIRST (80% Train, 20% Test)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Determine the sensitive column for l-diversity and t-closeness
    sensitive_col_for_privacy = target_col
    
    # 2. MASK SECOND (Apply ONLY to the Train Set, passing the domain_rules)
    train_k_anon = apply_k_anonymity(train_df, qi_cols, domain_rules, k=5)
    train_l_div = apply_l_diversity(train_df, qi_cols, sensitive_col_for_privacy, domain_rules, k=5, l=2)
    
    # --- ADDED: t-Closeness Implementation ---
    train_t_close = apply_t_closeness(train_df, qi_cols, sensitive_col_for_privacy, domain_rules, k=5, t=0.2)
    
    train_dp = apply_differential_privacy(train_df, num_sa_cols, epsilon=0.5)
    
    # 3. PREPROCESS ALL SETS
    train_raw_enc = preprocess_for_ml(train_df, target_col)
    train_k_enc = preprocess_for_ml(train_k_anon, target_col)
    train_l_enc = preprocess_for_ml(train_l_div, target_col)
    train_t_enc = preprocess_for_ml(train_t_close, target_col) # Added
    train_dp_enc = preprocess_for_ml(train_dp, target_col)
    
    test_raw_enc = preprocess_for_ml(test_df, target_col) # Test set remains mathematically RAW
    
    X_test = test_raw_enc.drop(columns=[target_col], errors='ignore')
    y_test = test_raw_enc[target_col] if target_col in test_raw_enc.columns else None
    
    results = {}
    
    # 4. ADDED t-Closeness TO THE EXECUTION PIPELINE
    datasets = {
        "Baseline (Raw)": train_raw_enc,
        "k-Anonymity": train_k_enc,
        "l-Diversity": train_l_enc,
        "t-Closeness": train_t_enc, 
        "Differential Privacy": train_dp_enc
    }
    
    # 5. TRAIN AND AUDIT
    for name, train_data in datasets.items():
        if len(train_data) == 0:
            results[name] = {"F1_Score": 0.0, "Bias_Score": 0.0, "Note": "Dropped all rows"}
            continue

        X_train = train_data.drop(columns=[target_col], errors='ignore')
        y_train = train_data[target_col] if target_col in train_data.columns else None
        
        common_cols = X_train.columns.intersection(X_test.columns)
        X_train_aligned = X_train[common_cols]
        X_test_aligned = X_test[common_cols]

        clf = RandomForestClassifier(random_state=42, n_estimators=50)
        clf.fit(X_train_aligned, y_train)
        
        y_pred = clf.predict(X_test_aligned)
        
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        if protected_col in X_test_aligned.columns:
            protected_test_vals = X_test_aligned[protected_col]
            try:
                bias_score = demographic_parity_difference(y_test, y_pred, sensitive_features=protected_test_vals)
            except ValueError:
                bias_score = 0.0 
        else:
            bias_score = 0.0
        
        results[name] = {
            "F1_Score": round(f1, 4),
            "Bias_Score": round(bias_score, 4)
        }
        
    return results