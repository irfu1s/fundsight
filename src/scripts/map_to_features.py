import pandas as pd
import re

features_path = "data/sip_features_model_ready.csv"
output_path   = "data/master_fund_map_final.csv"

print("🔹 Loading features file...")

df = pd.read_csv(features_path)

# 🛡️ SAFETY FIX 1: Robust Scheme Code Cleaning
# 1. Convert to string
# 2. Strip whitespace
# 3. Remove trailing '.0' if it exists (e.g. "10023.0" -> "10023")
df["scheme_code"] = df["scheme_code"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
df["scheme_name"] = df["scheme_name"].astype(str).str.strip()

print(f"   Loaded {len(df)} rows.")

# --- Match Key Builder ---
def create_match_key(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Normalize: Remove specific noise words
    text = re.sub(r'\b(direct|growth|regular|plan|fund|scheme|option|dividend|idcw|india)\b', '', text)
    # Remove all non-alphanumeric chars
    return re.sub(r'[^a-z0-9]', '', text)

print("🔹 Building identity map...")

df["canonical_scheme_name"] = df["scheme_name"]
df["match_key"] = df["scheme_name"].apply(create_match_key)

# Keep only identity columns
cols = [
    "scheme_code",
    "scheme_name",
    "canonical_scheme_name",
    "match_key",
]

# Preserve category if it exists (Critical for Advisor)
if "category" in df.columns:
    cols.append("category")

# 🛡️ SAFETY FIX 2: Strict Deduplication
# Ensure ONE row per scheme_code. We take the first occurrence.
df_map = df[cols].drop_duplicates(subset=["scheme_code"])

print(f"   Unique Funds Found: {len(df_map)}")

df_map.to_csv(output_path, index=False)

print(f"\n✅ Saved master identity map → {output_path}")
print("🎯 This is now your single source of truth.")