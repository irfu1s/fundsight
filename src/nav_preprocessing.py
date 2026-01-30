import pandas as pd
import re
from pathlib import Path

# ================= CONFIGURATION ================= #
# DO NOT CHANGE THESE FILENAMES (As per instruction)
RAW_PATH = Path("data/nav_history_raw.txt")
OUT_PATH = Path("data/nav_daily_clean.csv")

# Same noise words as Metadata Preprocessing for consistency
NOISE_WORDS = [
    r"\bdirect\b", r"\bregular\b", r"\bplan\b", r"\boption\b",
    r"\bgrowth\b", r"\bidcw\b", r"\bdividend\b", r"\bbonus\b",
    r"\bpay\s?out\b", r"\breinvestment\b", r"\bmonthly\b", r"\bquarterly\b",
    r"\bannual\b", r"\bseries\b", r"\b-\b"
]

# ================= HELPER FUNCTIONS ================= #

def normalize_scheme_name(name: str) -> str:
    """
    Standardizes the scheme name to match the Metadata file.
    Example: 'HDFC Small Cap Fund - Direct Growth' -> 'HDFC SMALL CAP FUND'
    """
    if not isinstance(name, str):
        return ""
    
    # Clean text
    clean = name.lower()
    for pattern in NOISE_WORDS:
        clean = re.sub(pattern, " ", clean)
    
    # Remove multiple spaces and strip
    clean = re.sub(r"\s+", " ", clean).strip().upper()
    return clean

def load_raw_history(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"❌ Raw NAV file not found: {path}")

    print(f"⏳ Loading raw NAV history from {path}...")
    df = pd.read_csv(
        path,
        sep=";",
        header=None,       # AMFI file has no header
        dtype=str,         # Load all as string first to prevent dtype warnings
        engine="python",
        on_bad_lines="skip"
    )
    print(f"   Loaded {len(df):,} rows.")
    return df

def clean_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    print("🧹 Cleaning and standardizing NAV data...")

    # 1) Assign AMFI standard 8 columns
    # We only care about code, name, nav, date for the SIP engine
    df.columns = [
        "scheme_code", "scheme_name", "isin_growth", "isin_div", 
        "nav", "repurchase_price", "sale_price", "date"
    ]

    # 2) Keep essentials
    df = df[["scheme_code", "scheme_name", "nav", "date"]].copy()

    # 3) Drop initial junk
    df = df.dropna(subset=["scheme_code", "nav", "date"])

    # 4) Basic String Cleaning
    df["scheme_code"] = df["scheme_code"].str.strip()
    df["raw_name"] = df["scheme_name"].str.strip() # Keep raw for reference if needed
    df["nav"] = df["nav"].str.strip()
    df["date"] = df["date"].str.strip()

    # 5) STRICT Name Normalization (Syncs with Metadata Step 1)
    df["scheme_name"] = df["raw_name"].apply(normalize_scheme_name)
    df = df.drop(columns=["raw_name"])

    # 6) Convert Types
    # NAV to Float
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    
    # Date to Datetime
    df["date"] = pd.to_datetime(
        df["date"],
        format="%d-%b-%Y",    # Example: 03-Jan-2017
        errors="coerce"
    )

    # 7) Drop failed conversions
    initial_rows = len(df)
    df = df.dropna(subset=["nav", "date"])
    dropped = initial_rows - len(df)
    if dropped > 0:
        print(f"   ⚠️ Dropped {dropped:,} rows due to invalid NAV/Date.")

    # 8) Sort & Deduplicate
    # Critical for Rolling Window calculations in Step 2
    df = (
        df.sort_values(["scheme_code", "date"])
          .drop_duplicates(subset=["scheme_code", "date"])
          .reset_index(drop=True)
    )

    return df

# ================= MAIN PIPELINE ================= #

def main():
    try:
        df_raw = load_raw_history()
        df_clean = clean_and_standardize(df_raw)

        # 🔍 Verification
        print("-" * 30)
        print("✅ NAV Cleaning Complete")
        print(f"   Final Rows: {len(df_clean):,}")
        print(f"   Unique Schemes: {df_clean['scheme_code'].nunique():,}")
        print(f"   Date Range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")
        print("-" * 30)

        # Sanity Sample
        print("\n📊 Sample Data:")
        print(df_clean.head(3))

        # Save
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(OUT_PATH, index=False)
        print(f"\n💾 Saved to: {OUT_PATH}")

    except Exception as e:
        print(f"\n❌ Error in NAV Preprocessing: {e}")

if __name__ == "__main__":
    main()