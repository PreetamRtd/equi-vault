from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
import io
import traceback

# Import our custom engines and utilities
from ml_auditor import run_audit
from attack_simulation import simulate_homogeneity_attack, simulate_skewness_attack
from utils import load_domain_rules, clean_json_types

app = FastAPI(title="Equi-Vault API")

@app.get("/")
def read_root():
    return {"status": "Equi-Vault Backend is Running. Ready to Audit."}

@app.post("/audit/")
async def audit_dataset(
    file: UploadFile = File(...),
    domain: str = Form(...),
    target_col: str = Form(...),
    protected_col: str = Form(...),
    qi_cols: str = Form(...), 
    sa_cols: str = Form(...)  
):
    try:
        # 1. Parse Inputs
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Safely split strings into lists
        qi_list = [col.strip() for col in qi_cols.split(",")] if qi_cols else []
        sa_list = [col.strip() for col in sa_cols.split(",")] if sa_cols else []
        
        # Determine numerical SAs for Differential Privacy
        num_sa_cols = [col for col in sa_list if pd.api.types.is_numeric_dtype(df[col])]
        
        # 2. Load Rules
        domain_rules = load_domain_rules(domain)
        
        # 3. Run the Machine Learning Audit
        ml_results = run_audit(df, target_col, protected_col, qi_list, num_sa_cols, domain_rules)
        
        # 4. Run the Attack Simulations (Vulnerability)
        from privacy_engine import apply_k_anonymity
        df_k_anon = apply_k_anonymity(df, qi_list, domain_rules, k=5)
        
        primary_sa = sa_list[0] if sa_list else target_col
        homogeneity_risk = simulate_homogeneity_attack(df_k_anon, qi_list, primary_sa)
        skewness_risk = simulate_skewness_attack(df_k_anon, qi_list, primary_sa)
        
        # 5. The Recommendation Engine
       # 5. The Recommendation Engine
        recommendation = "Baseline"
        
        # Get the scores
        baseline_f1 = ml_results.get("Baseline (Raw)", {}).get("F1_Score", 0)
        dp_f1 = ml_results.get("Differential Privacy", {}).get("F1_Score", 0)
        k_anon_f1 = ml_results.get("k-Anonymity", {}).get("F1_Score", 0)
        
        # Relative logic: Does DP keep 90% of the baseline's original accuracy?
        if dp_f1 >= (baseline_f1 * 0.90):
            recommendation = "Differential Privacy is highly recommended. It offers mathematical immunity to attacks while preserving high ML utility."
        elif homogeneity_risk.get("risk_percentage", 0) > 10:
            recommendation = "Hybrid (k-Anonymity + DP) is required to patch exposed rows."
        elif k_anon_f1 > dp_f1:
            recommendation = "k-Anonymity (k=5) is recommended. DP destroyed too much utility for this dataset."
        else:
            recommendation = "Differential Privacy optimally balances fairness, privacy, and utility for this dataset."

        # --- NEW: Generate Downloadable CSV based on Recommendation ---
        # 1. ALWAYS drop Direct Identifiers before exporting
        safe_df = df.drop(columns=['Name', 'name', 'Patient ID', 'id'], errors='ignore')

        # 2. Apply the winning technique to the safe dataframe
        if "Differential Privacy" in recommendation:
            from privacy_engine import apply_differential_privacy
            best_df = apply_differential_privacy(safe_df, num_sa_cols, epsilon=0.5)
        else:
            from privacy_engine import apply_k_anonymity
            best_df = apply_k_anonymity(safe_df, qi_list, domain_rules, k=5)
            
        # Convert the dataframe to a CSV string
        csv_buffer = io.StringIO()
        best_df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()

        # 6. Package Data
        response_payload = {
            "dataset_info": {
                "filename": file.filename,
                "total_rows": len(df),
                "domain": domain
            },
            "ml_audit": ml_results,
            "vulnerability_analysis": {
                "homogeneity_attack": homogeneity_risk,
                "skewness_attack": skewness_risk
            },
            "recommendation": recommendation,
            "downloadable_csv": csv_string # <--- Added this line
        }
        
        # 7. SANITIZE AND RETURN (This fixes the 500 error!)
        cleaned_payload = clean_json_types(response_payload)
        return JSONResponse(content=cleaned_payload)
        
    except Exception as e:
        # If it crashes, print the EXACT error to the terminal so we can see it
        print("\n--- CRASH REPORT ---")
        traceback.print_exc() 
        print("--------------------\n")
        return JSONResponse(status_code=500, content={"error": str(e)})