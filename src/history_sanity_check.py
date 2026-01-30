import pandas as pd
from pathlib import Path

# ================= CONFIGURATION ================= #
# Input: The output from Step 1 (NAV Preprocessing)
NAV_IN_PATH = Path("data/nav_daily_clean.csv")
# Output: The "High Quality" dataset for Step 2
NAV_OUT_PATH = Path("data/nav_daily_clean_filtered.csv")

# Quality Thresholds
MIN_YEARS = 8          # Require at least 8 distinct years of data (e.g., 2017-2025)
MIN_COVERAGE = 0.8     # Require at least 80% of months present (prevents big data gaps)
SAVE_FILTERED = True   # Set to True to actually save the file

# ================= MAIN LOGIC ================= #

def main():
    print(f"🔍 Starting Step 1.5: Data Quality Check...")
    print(f"   Loading NAV data from: {NAV_IN_PATH}")
    
    if not NAV_IN_PATH.exists():
        print(f"❌ Error: Input file not found at {NAV_IN_PATH}")
        return

    df = pd.read_csv(NAV_IN_PATH, parse_dates=["date"])
    df["scheme_code"] = df["scheme_code"].astype(str)

    # 1. Derive helper columns for stats
    df["year"] = df["date"].dt.year
    df["year_month"] = df["date"].dt.to_period("M")

    # 2. Group & Calculate Stats per Scheme
    print("   Calculating coverage stats per scheme...")
    grouped = df.groupby("scheme_code")

    stats = grouped.agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        n_years=("year", "nunique"),
        n_months=("year_month", "nunique"),
    )

    # Calculate theoretical months (Span) vs actual months (n_months)
    # This finds funds with "missing months"
    span_months = (
        stats["last_date"].dt.to_period("M") - stats["first_date"].dt.to_period("M")
    ).apply(lambda p: p.n + 1)

    stats["span_months"] = span_months
    stats["coverage_ratio"] = stats["n_months"] / stats["span_months"]

    # 3. Filter "Good" Schemes
    good_mask = (stats["n_years"] >= MIN_YEARS) & (stats["coverage_ratio"] >= MIN_COVERAGE)
    good_schemes = stats[good_mask].index.tolist()

    # 4. Report Results
    print("\n📊 Quality Summary:")
    print(f"   Total Schemes Input : {len(stats)}")
    print(f"   High Quality Schemes: {len(good_schemes)} (Passed)")
    print(f"   Rejected Schemes    : {len(stats) - len(good_schemes)} (Failed)")
    print(f"   Criteria            : >= {MIN_YEARS} years data AND >= {MIN_COVERAGE*100:.0f}% coverage")

    # Show some bad examples (for user sanity check)
    bad = stats[~good_mask].sort_values("coverage_ratio").head(5)
    if not bad.empty:
        print("\n❌ Examples of Rejected Schemes (Poor Coverage/History):")
        print(bad[["n_years", "coverage_ratio", "first_date", "last_date"]])

    # 5. Save the "Filtered" Dataset for Step 2
    if SAVE_FILTERED:
        df_filtered = df[df["scheme_code"].isin(good_schemes)].copy()
        
        # Cleanup helpers
        df_filtered = df_filtered.drop(columns=["year", "year_month"], errors="ignore")
        
        NAV_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_filtered.to_csv(NAV_OUT_PATH, index=False)
        print("-" * 40)
        print(f"✅ Saved Filtered NAV Data -> {NAV_OUT_PATH}")
        print(f"   Rows: {len(df_filtered):,}")
        print("-" * 40)
    else:
        print("\nℹ️ SAVE_FILTERED is False. No file was saved.")

if __name__ == "__main__":
    main()