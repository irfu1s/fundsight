import pandas as pd
import numpy as np
from pathlib import Path

# ================= CONFIGURATION ================= #
FEATURE_PATH = Path("data/sip_features_model_ready.csv")
PREDICTIONS_PATH = Path("data/sip_predictions.csv")

# ================= DATA LOADING ================= #

def normalize_scheme_name(s: str) -> str:
    """Normalize scheme_name to a base fund name by stripping noise."""
    if not isinstance(s, str): return ""
    x = s.upper()
    junk_words = [
        "DIRECT PLAN", "REGULAR PLAN", "GROWTH OPTION", "GROWTH",
        "IDCW", "DIVIDEND", "REINVESTMENT", "BONUS",
        "MONTHLY", "QUARTERLY", "ANNUAL", "PAYOUT", "PAY OUT",
        "OPTION", "PLAN", "-", "  ",
    ]
    for w in junk_words:
        x = x.replace(w, " ")
    return " ".join(x.split())

def load_data() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError("❌ Predictions file missing. Run model_inference.py first.")

    df = pd.read_csv(PREDICTIONS_PATH)
    df["scheme_code"] = df["scheme_code"].astype(str)
    
    # Load features if available (for 'is_active' checks)
    if FEATURE_PATH.exists():
        feats = pd.read_csv(FEATURE_PATH)
        feats["scheme_code"] = feats["scheme_code"].astype(str)
        cols_to_merge = ["scheme_code", "is_active", "min_sip_amount"]
        cols_to_merge = [c for c in cols_to_merge if c in feats.columns]
        df = df.merge(feats[cols_to_merge], on="scheme_code", how="left")

    return df

# ================= FILTERS ================= #

def filter_schemes(df: pd.DataFrame, category: str, risk_level: str) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Active Check
    if "is_active" in df.columns:
        df = df[df["is_active"] == True]

    # 2. Category Filter
    valid_cat = category.lower().replace(" ", "_")
    if valid_cat not in df["category"].unique():
        # Fallback mappings
        if "index" in valid_cat: valid_cat = "index_fund"
        elif "small" in valid_cat: valid_cat = "small_cap"
        elif "mid" in valid_cat: valid_cat = "mid_cap"
        elif "large" in valid_cat: valid_cat = "large_cap"
        elif "multi" in valid_cat: valid_cat = "multi_cap"
    
    df_cat = df[df["category"] == valid_cat]
    if df_cat.empty: return df 

    # 3. Risk Filter
    risk = risk_level.lower()
    if risk == "low":
        df_risk = df_cat[df_cat["category_risk_tier"].isin(["low", "medium"])]
    elif risk == "high":
        df_risk = df_cat[df_cat["category_risk_tier"].isin(["medium", "high"])]
    else:
        df_risk = df_cat[df_cat["category_risk_tier"] == "medium"]

    if df_risk.empty: return df_cat
    return df_risk

# ================= MATH ENGINES ================= #

def simulate_sip_logic(annual_cagr, monthly_amount, sip_years, hold_years, step_up_pct):
    """
    Complex Logic: Monthly Payments -> Step Up -> Optional Holding Phase
    """
    months_active = sip_years * 12
    rate_monthly = (1 + annual_cagr / 100) ** (1/12) - 1
    
    total_invested = 0
    current_value = 0
    current_sip = monthly_amount
    
    curve = []
    
    # Phase 1: Active SIP
    for m in range(1, months_active + 1):
        total_invested += current_sip
        current_value += current_sip
        current_value *= (1 + rate_monthly)
        
        # Apply Step-Up Annually
        if m % 12 == 0:
            current_sip = current_sip * (1 + step_up_pct / 100)
            curve.append({"year": m // 12, "invested": round(total_invested), "value": round(current_value)})

    # Phase 2: Optional Holding
    if hold_years > 0:
        for y in range(1, hold_years + 1):
            current_value *= (1 + annual_cagr / 100)
            curve.append({"year": sip_years + y, "invested": round(total_invested), "value": round(current_value)})
            
    return total_invested, current_value, curve

def simulate_lumpsum_logic(annual_cagr, total_amount, duration_years):
    """
    Simple Logic: One-time Payment -> Mandatory Holding (Duration) -> No Step-Up
    Formula: A = P(1 + r)^t
    """
    total_invested = total_amount
    
    # Calculate Final Value (Compound Interest)
    final_value = total_amount * ((1 + annual_cagr / 100) ** duration_years)
    
    # Generate Curve
    curve = []
    current_val = total_amount
    for y in range(1, duration_years + 1):
        current_val *= (1 + annual_cagr / 100)
        curve.append({"year": y, "invested": round(total_invested), "value": round(current_val)})
        
    return total_invested, final_value, curve

# ================= MAIN ADVISOR FUNCTION ================= #

def get_recommendations(
    category: str,
    risk_level: str,
    amount: float,           # Renamed: Context depends on type (Monthly or Total)
    duration_years: int,     # Renamed: SIP duration OR Lumpsum total duration
    investment_type: str = "sip", # <--- NEW ARGUMENT ('sip' or 'lumpsum')
    hold_years: int = 0,     # SIP Only (Optional)
    step_up_percent: float = 0.0, # SIP Only (Optional)
    top_k: int = 3
):
    # 1. Load Data
    df = load_data()
    
    # 2. Filter
    candidates = filter_schemes(df, category, risk_level)
    candidates = candidates.sort_values("predicted_cagr_percent", ascending=False)
    
    # 3. Deduplicate
    candidates["base_name"] = candidates["scheme_name"].apply(normalize_scheme_name)
    candidates = candidates.drop_duplicates(subset=["base_name"])
    
    results = []
    
    for idx, row in candidates.head(top_k).iterrows():
        cagr = row["predicted_cagr_percent"]
        
        # --- THE SWITCH ---
        if investment_type.lower() in ["lumpsum", "onetime", "one-time", "one time", "single"]:
            # Lumpsum Logic
            invested, final_val, curve = simulate_lumpsum_logic(cagr, amount, duration_years)
            lbl_type = "Lumpsum"
            disp_step_up = "N/A"
        else:
            # SIP Logic (Default)
            invested, final_val, curve = simulate_sip_logic(cagr, amount, duration_years, hold_years, step_up_percent)
            lbl_type = "SIP"
            disp_step_up = f"{step_up_percent}%"

        profit = final_val - invested
        
        results.append({
            "fund_name": row["scheme_name"],
            "category": row["category"],
            "risk": row["category_risk_tier"],
            "predicted_cagr": cagr,
            "type": lbl_type,
            "step_up": disp_step_up,
            "total_invested": round(invested),
            "final_value": round(final_val),
            "profit": round(profit),
            "projected_return_percent": (profit/invested)*100 if invested > 0 else 0,
            "curve": curve 
        })
        
    return results

# ================= TEST RUNNER ================= #
if __name__ == "__main__":
    print("--- Testing SIP (5yr Pay + 5yr Hold) ---")
    # Note: We now pass investment_type="sip"
    sip = get_recommendations("small_cap", "low", 5000, 5, "sip", hold_years=5, step_up_percent=10)[0]
    print(f"SIP Profit: ₹{sip['profit']:,}")

    print("\n--- Testing Lumpsum (10yr Duration) ---")
    # Note: We pass investment_type="lumpsum", hold_years/step_up ignored
    lump = get_recommendations("small_cap", "low", 50000, 10, "lumpsum")[0]
    print(f"Lumpsum Profit: ₹{lump['profit']:,}")