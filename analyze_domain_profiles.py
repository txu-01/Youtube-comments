import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
import os

# ========= 配置 =========
INPUT_CSV = "data/processed/all_domains_with_sentiment.csv"
OUTPUT_DIR = "figures/domain_profiles"
sns.set(style="whitegrid", font_scale=1.1)
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =======================

# 1. 读取数据
df = pd.read_csv(INPUT_CSV)
df["clean_text"] = df["clean_text"].astype(str).fillna("")

# -------- 列名兜底处理 --------
# 点赞列
if "like_count" not in df.columns:
    df["like_count"] = 0
else:
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)

# 评论长度列
if "comment_length" not in df.columns:
    if "char_len" in df.columns:
        df["comment_length"] = pd.to_numeric(df["char_len"], errors="coerce").fillna(0)
    else:
        df["comment_length"] = df["clean_text"].str.len()

# 置信度列
if "prob_positive" not in df.columns and "prob_pos" in df.columns:
    df["prob_positive"] = df["prob_pos"]
if "prob_negative" not in df.columns and "prob_neg" in df.columns:
    df["prob_negative"] = df["prob_neg"]
# -------------------------

domains = df["domain"].unique()

for dom in domains:
    sub = df[df["domain"] == dom].copy()
    print(f"\n===== {dom.upper()} =====")
    print(f"总评论数：{len(sub)}")

    # 2. 情感分布饼图
    plt.figure(figsize=(5,5))
    sub["sentiment"].value_counts().plot.pie(
        autopct='%1.1f%%', colors=["#E74C3C","#F1C40F","#2ECC71"],
        startangle=90, ylabel="")
    plt.title(f"Sentiment Distribution - {dom.capitalize()}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{dom}_sentiment_pie.png", dpi=200)
    plt.close()

    # 3. 情感强度分布直方图
    plt.figure(figsize=(6,4))
    sns.histplot(sub["prob_positive"], bins=20, color="green", label="Positive Prob", alpha=0.6)
    sns.histplot(sub["prob_negative"], bins=20, color="red", label="Negative Prob", alpha=0.4)
    plt.title(f"Sentiment Confidence - {dom.capitalize()}")
    plt.xlabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{dom}_sentiment_confidence.png", dpi=200)
    plt.close()

    # 4. 情感 × 点赞数
    plt.figure(figsize=(6,4))
    sns.boxplot(data=sub, x="sentiment", y="like_count",
                order=["negative","neutral","positive"])
    plt.yscale("log")
    plt.title(f"Like Count by Sentiment - {dom.capitalize()}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{dom}_like_by_sentiment.png", dpi=200)
    plt.close()

    # 5. 情感 × 评论长度
    plt.figure(figsize=(6,4))
    sns.boxplot(data=sub, x="sentiment", y="comment_length",
                order=["negative","neutral","positive"])
    plt.title(f"Comment Length by Sentiment - {dom.capitalize()}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{dom}_length_by_sentiment.png", dpi=200)
    plt.close()

    # 6. 打印正/负典型关键词
    def extract_words(texts):
        all_words = []
        for t in texts:
            words = re.findall(r"[A-Za-z']+", t.lower())
            all_words.extend(words)
        stopwords = set(["the","and","is","it","to","a","of","for","in","on","with","this","that","was","are","be"])
        words_clean = [w for w in all_words if w not in stopwords and len(w)>2]
        return Counter(words_clean).most_common(10)

    pos_words = extract_words(sub[sub["sentiment"]=="positive"]["clean_text"])
    neg_words = extract_words(sub[sub["sentiment"]=="negative"]["clean_text"])

    print("Top Positive Keywords:", pos_words)
    print("Top Negative Keywords:", neg_words)

    # 7. 词云（可选）
    for sent, words in [("positive",pos_words),("negative",neg_words)]:
        freq_dict = {w:c for w,c in words}
        if freq_dict:
            wc = WordCloud(width=600, height=400,
                           background_color="white",
                           colormap="Greens" if sent=="positive" else "Reds")
            wc.generate_from_frequencies(freq_dict)
            wc.to_file(f"{OUTPUT_DIR}/{dom}_{sent}_wordcloud.png")

print(f"\n分析完成！所有图表保存在 → {OUTPUT_DIR}/")
