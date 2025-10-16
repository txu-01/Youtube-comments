# merge_all_domains.py
import pandas as pd

files = {
    "sneaker": "data/processed/sneaker_comments_clean.csv",
    "pharma" : "data/processed/pharma_comments_clean.csv",
    "food"   : "data/processed/food_comments_clean.csv",
    "steam"  : "data/processed/steam_review_comments_2025_clean.csv"
}

dfs = []
for dom, path in files.items():
    df = pd.read_csv(path)
    df["domain"] = dom
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)
# 转换日期
if "published_at" in merged.columns:
    merged["published_at"] = pd.to_datetime(merged["published_at"], errors="coerce")

merged.to_csv("data/processed/all_domains_merged.csv", index=False)
print(f"合并完成，共 {len(merged)} 条评论，保存为 all_domains_merged.csv")
