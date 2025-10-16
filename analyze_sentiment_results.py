import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ========== 配置 ==========
INPUT_CSV = "data/processed/all_domains_with_sentiment.csv"
DATE_COL  = "published_at"
# ==========================

# 1. 读取数据
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows")

# 确保日期格式正确
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

# 检查关键列
print(df.head(3))

# 2. 基本统计
print("\n--- 评论数量（按领域） ---")
print(df["domain"].value_counts())

print("\n--- 情感分布（整体） ---")
print(df["sentiment"].value_counts())

print("\n--- 情感分布（按领域） ---")
print(pd.crosstab(df["domain"], df["sentiment"], normalize="index").round(3))

# 3. 可视化设置
sns.set(style="whitegrid", font_scale=1.1)

# 3.1 各领域情感分布柱状图
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="domain", hue="sentiment",
              order=df["domain"].value_counts().index)
plt.title("Sentiment Distribution by Domain")
plt.ylabel("Number of Comments")
plt.tight_layout()
plt.savefig("figures/sentiment_by_domain_bar.png", dpi=200)
plt.show()

# 3.2 各领域情感比例堆叠图
plt.figure(figsize=(8,5))
prop = pd.crosstab(df["domain"], df["sentiment"], normalize="index")
prop.plot(kind="bar", stacked=True, colormap="RdYlGn",
          figsize=(8,5))
plt.title("Proportion of Sentiment by Domain")
plt.ylabel("Proportion")
plt.tight_layout()
plt.savefig("figures/sentiment_by_domain_stacked.png", dpi=200)
plt.show()

# 3.3 时间趋势（按月）
df_time = df.copy()
df_time = df_time.dropna(subset=[DATE_COL])
df_time["month"] = df_time[DATE_COL].dt.to_period("M").astype(str)

monthly = (df_time.groupby(["month","domain","sentiment"])
           .size().reset_index(name="count"))

plt.figure(figsize=(10,6))
sns.lineplot(data=monthly[monthly["sentiment"]=="positive"],
             x="month", y="count", hue="domain", marker="o")
plt.xticks(rotation=45)
plt.title("Monthly Positive Sentiment Trends by Domain")
plt.tight_layout()
plt.savefig("figures/monthly_positive_trend.png", dpi=200)
plt.show()

# 4. 典型评论示例
print("\n--- 典型正向评论（每个领域各3条） ---")
for dom in df["domain"].unique():
    sample_pos = df[(df["domain"]==dom) & (df["sentiment"]=="positive")]\
                    .sort_values("prob_positive", ascending=False).head(3)
    print(f"\n[{dom.upper()} - Positive]")
    for t in sample_pos["clean_text"]:
        print(" •", t[:120])

print("\n--- 典型负向评论（每个领域各3条） ---")
for dom in df["domain"].unique():
    sample_neg = df[(df["domain"]==dom) & (df["sentiment"]=="negative")]\
                    .sort_values("prob_negative", ascending=False).head(3)
    print(f"\n[{dom.upper()} - Negative]")
    for t in sample_neg["clean_text"]:
        print(" •", t[:120])

print("\n图表已保存到 figures/ 目录。")
