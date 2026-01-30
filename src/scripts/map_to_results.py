import pandas as pd

results_path = "data/sip_results_active_clean.csv"
map_path     = "data/master_fund_map_final.csv"
output_path  = "data/sip_results_aligned.csv"

print("🔹 Loading datasets...")

df_res = pd.read_csv(results_path)
df_map = pd.read_csv(map_path)

# 🛡️ SAFETY 1: Normalize keys & remove '.0' decimals (Fixes Int vs Float mismatch)
df_res["scheme_code"] = df_res["scheme_code"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
df_map["scheme_code"] = df_map["scheme_code"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

print(f"   Results rows: {len(df_res)}")
print(f"   Map rows:     {len(df_map)}")

# Select identity columns from the Master Map
id_cols = ["scheme_code", "scheme_name", "canonical_scheme_name", "match_key"]
if "category" in df_map.columns:
    id_cols.append("category")

# 🛡️ SAFETY 2: Avoid Column Collision (_x, _y)
# We trust the Master Map more than the Results file for names/categories.
# Drop these columns from Results so the Map version takes over cleanly.
for col in id_cols:
    if col != "scheme_code" and col in df_res.columns:
        print(f"   ✂️ Dropping old '{col}' from results to accept master map version...")
        df_res.drop(columns=[col], inplace=True)

# Merge identity into results
df_merged = df_res.merge(
    df_map[id_cols],
    on="scheme_code",
    how="left"
)

# Stats & Validation
missing_count = df_merged["canonical_scheme_name"].isna().sum()
print("\n📊 Merge Stats")
print(f"   Total Rows:       {len(df_merged)}")
print(f"   Missing Matches:  {missing_count}")

if missing_count > 0:
    print("   ⚠ Note: Some funds have no match in the map. They will have NaN names.")

df_merged.to_csv(output_path, index=False)

print(f"\n✅ Saved aligned results → {output_path}")
print("🎯 Advisor + SHAP can now resolve fund names via match_key")