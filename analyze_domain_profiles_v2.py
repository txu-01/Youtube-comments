# analyze_domain_profiles_v2.py
# Domain-level profiling: sentiment mix, confidence, likes/length by sentiment, and top keywords.
# Input: settings.WITH_SENT_CSV (data2/processed/all_domains_with_sentiment.csv)
# Output figs: figures_v2/domain_profiles/*.png

from settings import WITH_SENT_CSV, FIG_DIR
import os, re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Optional wordcloud; if missing, just skip this feature
try:
    from wordcloud import WordCloud
    HAS_WC = True
except Exception:
    HAS_WC = False

INPUT_CSV = WITH_SENT_CSV
OUTPUT_DIR = FIG_DIR / "domain_profiles"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid", font_scale=1.05)

# ---- Load data
df = pd.read_csv(INPUT_CSV)
df["clean_text"] = df["clean_text"].astype(str).fillna("")
# Ensure required columns exist
if "comment_length" not in df.columns:
    df["comment_length"] = df["clean_text"].str.len()
if "like_count" in df.columns:
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)

# ---- Helper for top keywords
STOP = set("""the and is it to a of for in on with this that was are be you your from have not but can just when how why what who which its they them as at if so do we're i'm i've it's don't can't""".split())

def top_words(texts, k=15):
    allw = []
    for t in texts:
        ws = re.findall(r"[A-Za-z']+", str(t).lower())
        allw.extend(w for w in ws if w not in STOP and len(w) > 2)
    return Counter(allw).most_common(k)

domains = list(df["domain"].dropna().unique())

for dom in domains:
    sub = df[df["domain"] == dom].copy()
    print(f"\n=== {dom.upper()} ===  n={len(sub)}")

    # 1) Sentiment distribution
    plt.figure(figsize=(6,4))
    order = ["negative","neutral","positive"]
    counts = sub["sentiment"].value_counts().reindex(order)
    counts.plot(kind="bar", color=["#E74C3C","#F1C40F","#2ECC71"])
    plt.title(f"Sentiment Distribution - {dom}")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{dom}_sentiment_bar.png", dpi=200)
    plt.close()

    # 2) Confidence hist (positive/negative)
    if {"prob_positive","prob_negative"}.issubset(sub.columns):
        plt.figure(figsize=(6,4))
        sns.histplot(sub["prob_positive"], bins=30, label="Positive prob", alpha=0.6)
        sns.histplot(sub["prob_negative"], bins=30, label="Negative prob", alpha=0.6)
        plt.title(f"Sentiment Confidence - {dom}")
        plt.xlabel("Probability")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{dom}_confidence_hist.png", dpi=200)
        plt.close()

    # 3) Likes × Sentiment (log y)
    if "like_count" in sub.columns:
        plt.figure(figsize=(7,4))
        sns.boxplot(data=sub, x="sentiment", y="like_count", order=order)
        plt.yscale("log")
        plt.title(f"Like Count by Sentiment (log) - {dom}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{dom}_likes_by_sentiment.png", dpi=200)
        plt.close()

    # 4) Length × Sentiment
    plt.figure(figsize=(7,4))
    sns.boxplot(data=sub, x="sentiment", y="comment_length", order=order)
    plt.title(f"Comment Length by Sentiment - {dom}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{dom}_length_by_sentiment.png", dpi=200)
    plt.close()

    # 5) Top keywords (pos/neg) + optional wordclouds
    pos_kw = top_words(sub[sub["sentiment"]=="positive"]["clean_text"], k=15)
    neg_kw = top_words(sub[sub["sentiment"]=="negative"]["clean_text"], k=15)
    print("Top positive keywords:", pos_kw[:10])
    print("Top negative keywords:", pos_kw[:10])

    if HAS_WC:
        for sent, kws, cmap in [("positive", pos_kw, "Greens"), ("negative", neg_kw, "Reds")]:
            if kws:
                wc = WordCloud(width=900, height=600, background_color="white", colormap=cmap)
                wc.generate_from_frequencies({w:c for w,c in kws})
                wc.to_file(str(OUTPUT_DIR / f"{dom}_{sent}_wordcloud.png"))

print(f"\nDone. Figures → {OUTPUT_DIR}")
