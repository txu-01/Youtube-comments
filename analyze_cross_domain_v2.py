# analyze_cross_domain_v2.py
# Cross-domain comparison: sentiment proportions, extreme rates, engagement metrics, chi-square test.
# Input: settings.WITH_SENT_CSV
# Output figs: figures_v2/cross_domain/*.png

from settings import WITH_SENT_CSV, FIG_DIR
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

INPUT_CSV = WITH_SENT_CSV
OUTPUT_DIR = FIG_DIR / "cross_domain"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid", font_scale=1.05)

df = pd.read_csv(INPUT_CSV)
df["clean_text"] = df["clean_text"].astype(str).fillna("")
if "comment_length" not in df.columns:
    df["comment_length"] = df["clean_text"].str.len()
if "like_count" in df.columns:
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)

# 1) Sentiment proportion by domain (normalized)
prop = pd.crosstab(df["domain"], df["sentiment"], normalize="index")
print("\nSentiment proportion by domain:\n", prop.round(3))

plt.figure(figsize=(8,5))
prop.reindex(index=sorted(prop.index)).plot(kind="bar", stacked=True, colormap="RdYlGn", figsize=(9,5))
plt.title("Sentiment Proportions by Domain")
plt.ylabel("Proportion")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sentiment_proportion_by_domain.png", dpi=200)
plt.close()

# 2) Extreme sentiment rate (>0.9 on either pos or neg)
has_probs = {"prob_positive","prob_negative"}.issubset(df.columns)
if has_probs:
    df["extreme"] = ((df["prob_positive"]>0.9) | (df["prob_negative"]>0.9)).astype(int)
    extreme_prop = df.groupby("domain")["extreme"].mean().sort_values(ascending=False)
    print("\nExtreme sentiment rate by domain:\n", extreme_prop.round(3))

    plt.figure(figsize=(7,4))
    extreme_prop.plot(kind="bar", color="#8e44ad")
    plt.title("Extreme Sentiment Rate (>0.9)")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "extreme_sentiment_by_domain.png", dpi=200)
    plt.close()

# 3) Engagement metrics by domain × sentiment
order = ["negative","neutral","positive"]
if "like_count" in df.columns:
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x="domain", y="like_count", hue="sentiment", order=sorted(df["domain"].unique()), hue_order=order, ci=None)
    plt.yscale("log")
    plt.title("Average Like Count by Domain & Sentiment (log scale)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "likecount_by_domain_sentiment.png", dpi=200)
    plt.close()

plt.figure(figsize=(10,6))
sns.barplot(data=df, x="domain", y="comment_length", hue="sentiment", order=sorted(df["domain"].unique()), hue_order=order, ci=None)
plt.title("Average Comment Length by Domain & Sentiment")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "length_by_domain_sentiment.png", dpi=200)
plt.close()

# 4) Chi-square test (domain vs sentiment)
tab = pd.crosstab(df["domain"], df["sentiment"])
chi2, p, dof, exp = stats.chi2_contingency(tab)
print("\nChi-square test (domain × sentiment):")
print(f"Chi²={chi2:.2f}, dof={dof}, p-value={p:.3e}")
if p < 0.05:
    print("Conclusion: sentiment distribution differs significantly across domains.")
else:
    print("Conclusion: no significant difference across domains.")

print(f"\nDone. Figures → {OUTPUT_DIR}")
