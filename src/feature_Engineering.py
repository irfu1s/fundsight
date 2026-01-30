import pandas as pd
import numpy as np
from pathlib import Path

# ================= CONFIGURATION ================= #
# Inputs
METADATA_PATH = Path("data/cleaned_fund_data.csv")           # Output of Step 1
NAV_PATH = Path("data/nav_daily_clean_filtered.csv")         # Output of Step 1.5
SIP_RESULTS_PATH = Path("data/sip_results_active_clean.csv") # Output of SIP Engine

# Output (The "Truth" File for Model & Recommender)
OUT_PATH = Path("data/sip_features_model_ready.csv")

# Constants
RISK_FREE_RATE = 0.06  # 6% for Sharpe Ratio

# ================= HELPER FUNCTIONS ================= #

def safe_div(a, b):
    """
    ✅ CRITICAL FIX: Divides a by b safely.
    Returns 0 if b is 0, NaN, or Infinity.
    Prevents 'Infinity' errors that crash SHAP and Scalers.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(a, b)
        # Replace Inf, -Inf, and NaN with 0
        result = np.nan_to_num(result, posinf=0.0, neginf=0.0, nan=0.0)
    return result

def compute_volatility_and_sharpe(nav_df: pd.DataFrame, years: int = 3) -> pd.DataFrame:
    """
    Computes annualized volatility and Sharpe Ratio using daily returns.
    Limits data to the last `years` to keep stats relevant.
    """
    print(f"   Calculating Volatility & Sharpe (Last {years} Years)...")
    
    # 1. Setup Dates
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    cutoff_date = nav_df["date"].max() - pd.DateOffset(years=years)
    
    # 2. Filter Recent Data
    recent = nav_df[nav_df["date"] >= cutoff_date].copy()
    
    # 3. Calculate Daily Returns
    recent = recent.sort_values(["scheme_code", "date"])
    recent["daily_return"] = recent.groupby("scheme_code")["nav"].pct_change()
    
    # 4. Aggregation
    stats = recent.groupby("scheme_code")["daily_return"].agg(["std", "mean"])
    
    # 5. Annualize
    # Volatility = Daily Std * sqrt(252 trading days)
    stats["daily_return_std"] = stats["std"] * np.sqrt(252)
    
    # Annualized Return ~= Mean Daily Return * 252
    annualized_return = stats["mean"] * 252
    
    # ✅ FIX: Use safe_div for Sharpe Ratio
    stats["sharpe_ratio"] = safe_div((annualized_return - RISK_FREE_RATE), stats["daily_return_std"])
    
    # Clean up
    return stats[["daily_return_std", "sharpe_ratio"]].reset_index()

def compute_max_drawdown(nav_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the worst peak-to-valley drop.
    """
    print("   Calculating Max Drawdown...")
    
    def calculate_dd(series):
        roll_max = series.cummax()
        # safe_div not strictly needed here as roll_max shouldn't be 0 for NAV, but good practice
        dd = (series - roll_max) / roll_max 
        return dd.min()

    mdd = nav_df.groupby("scheme_code")["nav"].apply(calculate_dd).rename("max_drawdown")
    return mdd.reset_index()

def apply_relative_risk_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates volatility percentile STRICTLY within each category.
    """
    print("   Applying Relative Risk Ranking (Peer Comparison)...")
    
    # Drop rows with no category
    df = df.dropna(subset=["category"])
    
    # 1. Compute Percentile Rank (0.0 to 1.0) for Volatility inside Category
    df["category_volatility_pct"] = df.groupby("category")["daily_return_std"].rank(pct=True)
    
    # 2. Assign Tiers
    def assign_tier(pct):
        if pd.isna(pct): return "unknown"
        if pct <= 0.25: return "low"      # Bottom 25% Volatility
        if pct <= 0.75: return "medium"   # Middle 50%
        return "high"                     # Top 25% Volatility
        
    df["category_risk_tier"] = df["category_volatility_pct"].apply(assign_tier)
    
    return df

# ================= MAIN PIPELINE ================= #

def build_features():
    print("🚀 Starting Step 2: Feature Engineering (Safe Mode)...")

    # 1. Load Data
    if not METADATA_PATH.exists():
        print(f"❌ Error: Metadata file not found at {METADATA_PATH}")
        return
    if not NAV_PATH.exists():
        print(f"❌ Error: NAV file not found at {NAV_PATH}")
        return

    meta = pd.read_csv(METADATA_PATH) 
    nav = pd.read_csv(NAV_PATH)
    sip = pd.read_csv(SIP_RESULTS_PATH)

    # Ensure Codes are Strings
    meta["scheme_code"] = meta["scheme_code"].astype(str)
    nav["scheme_code"] = nav["scheme_code"].astype(str)
    sip["scheme_code"] = sip["scheme_code"].astype(str)

    # 2. Filter NAVs
    valid_schemes = meta["scheme_code"].unique()
    nav_filtered = nav[nav["scheme_code"].isin(valid_schemes)].copy()

    # 3. Compute Financial Stats
    vol_stats = compute_volatility_and_sharpe(nav_filtered, years=3)
    dd_stats = compute_max_drawdown(nav_filtered)

    # 4. Merge Everything
    df = sip.merge(meta[["scheme_code", "scheme_name", "category"]], on="scheme_code", how="inner")
    df = df.merge(vol_stats, on="scheme_code", how="left")
    df = df.merge(dd_stats, on="scheme_code", how="left")

    # 5. Compute Relative Risk
    df = apply_relative_risk_ranking(df)

    # 6. Final Clean & Save
    required_cols = ["category", "daily_return_std", "category_risk_tier", "cagr_percent"]
    df_clean = df.dropna(subset=required_cols).copy()

    # Log Transforms
    if "total_invested" in df_clean.columns:
        df_clean["log_total_invested"] = np.log1p(df_clean["total_invested"])

    # ✅ FINAL SAFETY CHECK: Replace any remaining Infinity/NaN with 0
    df_clean = df_clean.replace([np.inf, -np.inf], 0)
    df_clean = df_clean.fillna(0)
        
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUT_PATH, index=False)
    
    print("-" * 40)
    print("✅ Feature Engineering Complete!")
    print(f"   Saved to: {OUT_PATH}")
    print(f"   Total Funds Ready: {len(df_clean)}")
    print("-" * 40)
    
    # Sanity Check
    print("\n📊 Risk Tier Distribution per Category:")
    print(df_clean.groupby(["category", "category_risk_tier"]).size().unstack(fill_value=0))

if __name__ == "__main__":
    build_features()