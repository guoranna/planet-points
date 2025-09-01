import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz

# ---------------- 0. Parameters ----------------
THRESH      = 70
SALE_FILE   = "UCL Export 2.csv"                 # original dataset
PROD_FILE   = "PlanetPoints-Datasets_products_with_points.csv"
FINAL_FILE  = "filtered_UCL_data_with_scores.csv"

# ---------------- helpers ----------------
def _norm(s: str) -> str:
    return s.lower().replace(" ", "").replace("_", "")

def safe_cols(df, cols):
    return [c for c in cols if c in df.columns]

# ---------------- 1. Product library ----------------
df_prod = pd.read_csv(PROD_FILE, encoding="cp1252") 

# 1.1 unify key
df_prod["key"] = df_prod["name"].astype(str).str.strip().str.lower()

# 1.2 impactRating (1–5) -> A–E
if "impactRating" not in df_prod.columns:
    raise KeyError("Column 'impactRating' not found in product file.")
df_prod["impactRating"] = (
    pd.to_numeric(df_prod["impactRating"], errors="coerce")
      .map({1:"A", 2:"B", 3:"C", 4:"D", 5:"E"})
)

# 1.3 find reward points column (common variants)
reward_candidates = [c for c in df_prod.columns
                     if _norm(c) in {"rewardpoints","rewardpoint","pointsreward"}]
reward_col = reward_candidates[0] if reward_candidates else None
if reward_col is None:
    reward_col = "rewardPoints"
    df_prod[reward_col] = np.nan
    print("cannot find reward points column,An empty 'rewardPoints' column has been created for output. ")


prod_keep = safe_cols(
    df_prod,
    ["key", "emissionScore", "itemEmissionScore", "impactRating"]
)
if reward_col in df_prod.columns:
    prod_keep.append(reward_col)

# ---------------- 2. Sales flow: Filter out unnecessary lines ----------------
df_sales = pd.read_csv(SALE_FILE, encoding="cp1252")
mask     = df_sales["sDescription"].isin(["Breakfast", "Zero Value", "CARD", "Dinner"])
df_sales = df_sales.loc[~mask].copy()

# unify key
df_sales["key"] = df_sales["sDescription"].astype(str).str.strip().str.lower()

# precise merge first
df_merged = df_sales.merge(df_prod[prod_keep], on="key", how="left")

# ---------------- 3. Fuzzy backfill for unmatched ----------------
unmatched_keys = (
    df_merged.loc[df_merged["emissionScore"].isna(), "key"]
             .dropna().unique()
)
product_keys = df_prod["key"].tolist()

matches = []
for k in unmatched_keys:
    m = process.extractOne(k, product_keys,
                           scorer=fuzz.token_sort_ratio,
                           score_cutoff=THRESH)
    if m:
        matches.append({"key": k, "fuzzy_key": m[0], "match_score": m[1]})

df_fuzzy = pd.DataFrame(matches)

if not df_fuzzy.empty:
    df_fuzzy = df_fuzzy.merge(
        df_prod[prod_keep],
        left_on="fuzzy_key", right_on="key",
        how="left", suffixes=("", "_prod")
    )
    
    patch_cols = ["key"] + [c for c in ["emissionScore","itemEmissionScore","impactRating", reward_col]
                            if c in df_fuzzy.columns]
    patch = df_fuzzy[patch_cols]

    df_merged = df_merged.merge(patch, on="key", how="left", suffixes=("", "_p"))

    
    for col in ["emissionScore","itemEmissionScore","impactRating", reward_col]:
        if col in df_merged.columns and f"{col}_p" in df_merged.columns:
            df_merged[col] = df_merged[col].combine_first(df_merged[f"{col}_p"])

   
    drop_cols = [c for c in [f"{c}_p" for c in ["emissionScore","itemEmissionScore","impactRating", reward_col]]
                 if c in df_merged.columns]
    df_merged = df_merged.drop(columns=drop_cols)


df_merged = df_merged.drop(columns=["key"])

# ---------------- 4. Output ----------------
df_merged.to_csv(FINAL_FILE, index=False)

print(
    f" Generate {FINAL_FILE}\n"
    f" Unmatched rows (emissionScore is NaN): {df_merged['emissionScore'].isna().sum()}"
)
