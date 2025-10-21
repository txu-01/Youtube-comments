# merge_all_domains_v2.py
# Merge all cleaned domain CSVs into one file in data2/processed.

from settings import PROCESSED_DIR, MERGED_CSV
import pandas as pd

FILES = [
    "sneaker_comments_clean.csv",
    "pharma_comments_clean.csv",
    "food_comments_clean.csv",
    "steam_review_comments_clean.csv",
]

dfs = []
for name in FILES:
    path = PROCESSED_DIR / name
    df = pd.read_csv(path)
    # ensure domain exists
    if "domain" not in df.columns:
        raise ValueError(f"Missing domain column in {path}")
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)
if "published_at" in merged.columns:
    merged["published_at"] = pd.to_datetime(merged["published_at"], errors="coerce")
merged.to_csv(MERGED_CSV, index=False)
print(f"Merged {len(merged)} rows → {MERGED_CSV}")
