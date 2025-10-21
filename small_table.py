import pandas as pd
from settings import WITH_SENT_CSV, FIG_DIR

df = pd.read_csv(WITH_SENT_CSV)
prop = pd.crosstab(df["domain"], df["sentiment"], normalize="index").round(3)
prop.to_csv(FIG_DIR / "cross_domain" / "sentiment_proportions_table.csv")

ext = ((df["prob_positive"]>0.9)|(df["prob_negative"]>0.9)).groupby(df["domain"]).mean().round(3)
ext.to_csv(FIG_DIR / "cross_domain" / "extreme_rate_table.csv")
