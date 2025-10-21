# clean_comments_v2.py
# Clean raw CSVs from data2/raw → data2/processed, unify schema.

from settings import RAW_DIR, PROCESSED_DIR
import pandas as pd
import re

IN_OUT = {
    "sneaker": ("sneaker_comments_raw.csv", "sneaker_comments_clean.csv"),
    "pharma" : ("pharma_comments_raw.csv",  "pharma_comments_clean.csv"),
    "food"   : ("food_comments_raw.csv",    "food_comments_clean.csv"),
    "steam"  : ("steam_comments_raw.csv",   "steam_review_comments_clean.csv"),
}

def clean_text(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"`{1,3}.*?`{1,3}", " ", s)           # code spans
    s = re.sub(r"https?://\S+", " ", s)              # urls
    s = re.sub(r"[@#]\w+", " ", s)                   # @mentions, #tags
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_one(domain, in_name, out_name):
    src = RAW_DIR / in_name
    dst = PROCESSED_DIR / out_name
    df = pd.read_csv(src)

    # standardize columns if exist
    keep_cols = {
        "video_id": "video_id",
        "video_title": "video_title",
        "video_published_at": "video_published_at",
        "comment_id": "comment_id",
        "published_at": "published_at",
        "like_count": "like_count",
        "text": "text",
        "domain": "domain",
    }
    for k in list(keep_cols.keys()):
        if k not in df.columns:
            # tolerate missing, will fill later
            pass

    # fill domain if missing
    if "domain" not in df.columns:
        df["domain"] = domain

    # clean text
    df["text"] = df["text"].astype(str).fillna("")
    df["clean_text"] = df["text"].map(clean_text)

    # comment length
    df["comment_length"] = df["clean_text"].str.len()

    # coerce dtypes
    if "like_count" in df.columns:
        df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype("int32")

    # final column order (keep those that exist)
    cols = ["video_id","video_title","video_published_at",
            "comment_id","published_at","like_count",
            "clean_text","comment_length","domain"]
    df = df[[c for c in cols if c in df.columns]].dropna(subset=["clean_text"])
    df = df[df["clean_text"].str.strip().astype(bool)]

    df.to_csv(dst, index=False)
    print(f"[{domain}] Cleaned → {dst} ({len(df)} rows)")

if __name__ == "__main__":
    for dom, (i,o) in IN_OUT.items():
        clean_one(dom, i, o)
