import pandas as pd
import re
from pathlib import Path

# ================= CONFIG ================= #
NAV_PATH = Path("data/nav_daily_clean_filtered.csv")

OUTPUT_PATH = Path("data/cleaned_fund_data.csv")
REJECTED_LOG_PATH = Path("data/rejected_funds.csv")

# NEW — canonical identity mapping file
MASTER_MAP_PATH = Path("data/master_fund_map.csv")


# ================= HELPERS ================= #

def normalize_scheme_name(name: str) -> str:
    if not isinstance(name, str):
        return ""

    noise = [
        r"\bdirect\b", r"\bregular\b", r"\bplan\b", r"\boption\b",
        r"\bgrowth\b", r"\bidcw\b", r"\bdividend\b", r"\bbonus\b",
        r"\bpay\s?out\b", r"\breinvestment\b", r"\bmonthly\b",
        r"\bquarterly\b", r"\bannual\b", r"\bseries\b", r"\b-\b"
    ]

    clean = name.lower()
    for pattern in noise:
        clean = re.sub(pattern, " ", clean)

    clean = re.sub(r"\s+", " ", clean).strip()

    # canonical display form
    return clean.title()


def create_match_key(name: str) -> str:
    """Same key logic used in Advisor & Recommender"""
    if not isinstance(name, str):
        return ""
    text = name.lower()
    text = re.sub(
        r'\b(direct|growth|regular|plan|fund|scheme|option|dividend|idcw|india)\b',
        '',
        text
    )
    return re.sub(r'[^a-z0-9]', '', text)


def assign_strict_category(name: str) -> str:
    n = name.upper()

    if any(x in n for x in ["NIFTY", "SENSEX", "INDEX FUND", "ETF", "PASSIVE"]):
        return "index_fund"
    if "SMALL CAP" in n or "SMALLCAP" in n:
        return "small_cap"
    if "MID CAP" in n or "MIDCAP" in n:
        return "mid_cap"
    if "LARGE CAP" in n or "LARGECAP" in n or "BLUECHIP" in n:
        return "large_cap"
    if any(x in n for x in ["MULTI CAP", "MULTICAP", "FLEXI CAP", "FLEXICAP", "LARGE & MID"]):
        return "multi_cap"

    return "unknown"


# ================= MAIN ================= #

def run_preprocessing():
    print("🚀 Step 1: Metadata + Mapping Extraction")

    if not NAV_PATH.exists():
        print(f"❌ Error: Input file {NAV_PATH} not found.")
        return

    print(f"📥 Reading {NAV_PATH} ...")
    df_nav = pd.read_csv(NAV_PATH)

    df = df_nav[["scheme_code", "scheme_name"]].drop_duplicates().copy()
    print(f"🔹 Found {len(df)} unique funds")

    # canonical base name
    df["canonical_scheme_name"] = df["scheme_name"].apply(normalize_scheme_name)

    # match key used everywhere
    df["match_key"] = df["canonical_scheme_name"].apply(create_match_key)

    # category (strict)
    df["category"] = df["scheme_name"].apply(assign_strict_category)

    df["is_active"] = True

    valid = {"small_cap", "mid_cap", "large_cap", "multi_cap", "index_fund"}
    clean_df = df[df["category"].isin(valid)].copy()
    rejected_df = df[~df["category"].isin(valid)].copy()
    rejected_df["rejection_reason"] = "Unknown Category"

    # =========================
    # SAVE NORMAL OUTPUT (same)
    # =========================
    clean_df.to_csv(OUTPUT_PATH, index=False)
    rejected_df.to_csv(REJECTED_LOG_PATH, index=False)

    # =========================
    # SAVE MASTER FUND MAP
    # =========================
    master_cols = [
        "scheme_code",
        "scheme_name",
        "canonical_scheme_name",
        "match_key",
        "category"
    ]

    clean_df[master_cols].to_csv(MASTER_MAP_PATH, index=False)

    print("-----------------------------------------")
    print("✅ Metadata Extracted")
    print(f"✔ Clean funds saved to: {OUTPUT_PATH}")
    print(f"✔ Mapping saved to:     {MASTER_MAP_PATH}")
    print(f"⚠ Rejected: {len(rejected_df)} (logged)")
    print("-----------------------------------------")

    print("\nCategory counts:")
    print(clean_df["category"].value_counts())


if __name__ == "__main__":
    run_preprocessing()
