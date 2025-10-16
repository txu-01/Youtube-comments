import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

# ========= 配置 =========
INPUT_CSV = "data/processed/all_domains_with_sentiment.csv"
OUTPUT_DIR = "figures/cross_domain"
sns.set(style="whitegrid", font_scale=1.1)
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =======================

# 1. 读取数据
df = pd.read_csv(INPUT_CSV)
df["clean_text"] = df["clean_text"].astype(str).fillna("")
df["like_count"] = pd.to_numeric(df.get("like_count", 0), errors="coerce").fillna(0)
if "comment_length" not in df.columns:
    if "char_len" in df.columns:
        df["comment_length"] = pd.to_numeric(df["char_len"], errors="coerce").fillna(0)
    else:
        df["comment_length"] = df["clean_text"].str.len()

# 2. 各领域情感比例对比
prop = pd.crosstab(df["domain"], df["sentiment"], normalize="index")
print("\n--- 各领域情感比例 ---")
print(prop.round(3))

plt.figure(figsize=(8,5))
prop.plot(kind="bar", stacked=True, colormap="RdYlGn", figsize=(8,5))
plt.title("Sentiment Proportions by Domain")
plt.ylabel("Proportion")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sentiment_proportion_by_domain.png", dpi=200)
plt.close()

# 3. 极端情绪比例 (置信度>0.9)
if "prob_positive" in df.columns and "prob_negative" in df.columns:
    df["extreme"] = ((df["prob_positive"]>0.9) | (df["prob_negative"]>0.9)).astype(int)
    extreme_prop = df.groupby("domain")["extreme"].mean()
    print("\n--- 极端情绪比例 ---")
    print(extreme_prop.round(3))

    plt.figure(figsize=(6,4))
    extreme_prop.plot(kind="bar", color="purple")
    plt.title("Proportion of Extreme Sentiment (>0.9 confidence)")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/extreme_sentiment_by_domain.png", dpi=200)
    plt.close()

# 4. 点赞数 & 评论长度（按情感对比，取平均值）
agg = df.groupby(["domain","sentiment"]).agg({
    "like_count":"mean",
    "comment_length":"mean"
}).round(1)
print("\n--- 各领域互动指标均值 ---")
print(agg)

plt.figure(figsize=(10,6))
sns.barplot(data=df, x="domain", y="like_count", hue="sentiment", ci=None, order=df["domain"].unique())
plt.yscale("log")
plt.title("Average Like Count by Domain & Sentiment (log scale)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/likecount_by_domain_sentiment.png", dpi=200)
plt.close()

plt.figure(figsize=(10,6))
sns.barplot(data=df, x="domain", y="comment_length", hue="sentiment", ci=None, order=df["domain"].unique())
plt.title("Average Comment Length by Domain & Sentiment")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/length_by_domain_sentiment.png", dpi=200)
plt.close()

# 5. 卡方检验（领域 vs 情感）
table = pd.crosstab(df["domain"], df["sentiment"])
chi2, p, dof, exp = stats.chi2_contingency(table)
print("\n--- 卡方检验 ---")
print("Chi² =", round(chi2,2), "p-value =", p)
if p < 0.05:
    print("结论：不同领域的情感分布差异显著。")
else:
    print("结论：不同领域的情感分布差异不显著。")

print(f"\n分析完成！图表已保存到 → {OUTPUT_DIR}/")
