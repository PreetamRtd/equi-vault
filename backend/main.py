from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
import io
import traceback

# Import our custom engines and utilities
from ml_auditor import run_audit
from attack_simulation import simulate_homogeneity_attack, simulate_skewness_attack
from utils import get_dynamic_rules, clean_json_types

# Ensure all your privacy functions are imported!
from privacy_engine import apply_k_anonymity, apply_differential_privacy
# Fallbacks in case l-diversity/t-closeness dataset generators aren't fully written yet
try:
    from privacy_engine import apply_l_diversity, apply_t_closeness
except ImportError:
    pass

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
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        qi_list = [col.strip() for col in qi_cols.split(",")] if qi_cols else []
        sa_list = [col.strip() for col in sa_cols.split(",")] if sa_cols else []
        num_sa_cols = [col for col in sa_list if pd.api.types.is_numeric_dtype(df[col])]
        
        domain_rules = get_dynamic_rules(df, qi_list)
        ml_results = run_audit(df, target_col, protected_col, qi_list, num_sa_cols, domain_rules)
        
        df_k_anon = apply_k_anonymity(df, qi_list, domain_rules, k=5)
        primary_sa = sa_list[0] if sa_list else target_col
        homogeneity_risk = simulate_homogeneity_attack(df_k_anon, qi_list, primary_sa)
        skewness_risk = simulate_skewness_attack(df_k_anon, qi_list, primary_sa)
        
        # --- THE TOURNAMENT ENGINE ---
        baseline_f1 = ml_results.get("Baseline (Raw)", {}).get("F1_Score", 0.001) # Avoid div by zero
        utility_threshold = baseline_f1 * 0.80 

        candidates = ["k-Anonymity", "l-Diversity", "t-Closeness", "Differential Privacy"]
        viable_candidates = []
        
        for tech in candidates:
            stats = ml_results.get(tech, {})
            if not stats: continue
            
            f1 = stats.get("F1_Score", 0)
            bias = stats.get("Bias_Score", 1) 
            
            if f1 >= utility_threshold:
                viable_candidates.append({"name": tech, "f1": f1, "bias": bias})
                
        if viable_candidates:
            best_candidate = min(viable_candidates, key=lambda x: x["bias"])
            winner_name = best_candidate["name"]
            recommendation = f"🏆 {winner_name} is the clear winner! It retained high ML accuracy ({best_candidate['f1']:.2f}) while minimizing demographic bias to {best_candidate['bias']:.4f}."
        else:
            best_candidate = max([{"name": t, "f1": ml_results.get(t, {}).get("F1_Score", 0), "bias": ml_results.get(t, {}).get("Bias_Score", 1)} for t in candidates if t in ml_results], key=lambda x: x["f1"])
            winner_name = best_candidate["name"]
            recommendation = f"⚠️ Utility Warning: All techniques caused a drop in ML accuracy. {winner_name} was selected as the best compromise, preserving the most utility ({best_candidate['f1']:.2f})."

        # --- EXPORT THE WINNER ---
        safe_df = df.drop(columns=['Name', 'name', 'Patient ID', 'id'], errors='ignore')

        if winner_name == "Differential Privacy":
            best_df = apply_differential_privacy(safe_df, num_sa_cols, epsilon=0.5)
        elif winner_name == "l-Diversity" and 'apply_l_diversity' in globals():
            best_df = apply_l_diversity(safe_df, qi_list, primary_sa, domain_rules, l=2)
        elif winner_name == "t-Closeness" and 'apply_t_closeness' in globals():
            best_df = apply_t_closeness(safe_df, qi_list, primary_sa, domain_rules, t=0.2)
        else:
            best_df = apply_k_anonymity(safe_df, qi_list, domain_rules, k=5)
            
        csv_buffer = io.StringIO()
        best_df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()

        response_payload = {
            "dataset_info": {"filename": file.filename, "total_rows": len(df), "domain": domain},
            "ml_audit": ml_results,
            "vulnerability_analysis": {"homogeneity_attack": homogeneity_risk, "skewness_attack": skewness_risk},
            "recommendation": recommendation,
            "winner_name": winner_name,
            "downloadable_csv": csv_string
        }
        
        cleaned_payload = clean_json_types(response_payload)
        return JSONResponse(content=cleaned_payload)
        
    except Exception as e:
        print("\n--- CRASH REPORT ---")
        traceback.print_exc() 
        print("--------------------\n")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/audit_phase2/")
async def audit_phase2(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    protected_col: str = Form(...),
    qi_cols: str = Form(...), 
    sa_cols: str = Form(...),
    technique: str = Form(...)  
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        qi_list = [col.strip() for col in qi_cols.split(",")] if qi_cols else []
        sa_list = [col.strip() for col in sa_cols.split(",")] if sa_cols else []
        num_sa_cols = [col for col in sa_list if pd.api.types.is_numeric_dtype(df[col])]
        primary_sa = sa_list[0] if sa_list else target_col
        
        domain_rules = get_dynamic_rules(df, qi_list)
        
        # 1. Evaluate the ML impact of stacking
        ml_results = run_audit(df, target_col, protected_col, qi_list, num_sa_cols, domain_rules)
        phase1_baseline = ml_results.get("Baseline (Raw)", {})
        phase2_stats = ml_results.get(technique, {})
        
        # 2. Dynamically Apply Phase 2 Technique
        safe_df = df.copy()
        if technique == "Differential Privacy":
            from privacy_engine import apply_differential_privacy
            best_df = apply_differential_privacy(safe_df, num_sa_cols, epsilon=0.5)
        elif technique == "l-Diversity" and 'apply_l_diversity' in globals():
            from privacy_engine import apply_l_diversity
            best_df = apply_l_diversity(safe_df, qi_list, primary_sa, domain_rules, l=2)
        elif technique == "t-Closeness" and 'apply_t_closeness' in globals():
            from privacy_engine import apply_t_closeness
            best_df = apply_t_closeness(safe_df, qi_list, primary_sa, domain_rules, t=0.2)
        else:
            from privacy_engine import apply_k_anonymity
            best_df = apply_k_anonymity(safe_df, qi_list, domain_rules, k=5)
            
        # 3. Calculate vulnerabilities on the new stacked dataset
        homogeneity_risk = simulate_homogeneity_attack(best_df, qi_list, primary_sa)
        skewness_risk = simulate_skewness_attack(best_df, qi_list, primary_sa)
        
        csv_buffer = io.StringIO()
        best_df.to_csv(csv_buffer, index=False)
        
        # Format ML data so the frontend can plot Phase 1 vs Stacked Phase 2
        phase2_ml_audit = {
            "Phase 1 Data": phase1_baseline,
            f"+ {technique} (Stacked)": phase2_stats
        }
        
        response_payload = {
            "phase1_baseline": phase1_baseline,
            "phase2_results": phase2_stats,
            "ml_audit": phase2_ml_audit,
            "vulnerability_analysis": {"homogeneity_attack": homogeneity_risk, "skewness_attack": skewness_risk},
            "dataset_info": {"total_rows": len(best_df)},
            "technique": technique,
            "downloadable_csv": csv_buffer.getvalue()
        }
        
        return JSONResponse(content=clean_json_types(response_payload))
        
    except Exception as e:
        print("\n--- CRASH REPORT ---")
        traceback.print_exc() 
        return JSONResponse(status_code=500, content={"error": str(e)})