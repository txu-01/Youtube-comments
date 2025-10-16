import re
import os
import pandas as pd
from typing import Optional

# ========== 配置 ==========
INPUT_CSV  = "data/raw/food_comments.csv"         # 改成你的输入
OUTPUT_CSV = "data/processed/food_comments_clean.csv"
ENSURE_LANG = None      # None / "en" / "zh"；需安装 langdetect 才生效
APPLY_DOMAIN_FILTER = False  # 是否启用领域关键词过滤
DOMAIN_KEYWORDS = [
    # 示例：球鞋
    r"sneaker", r"jordan", r"nike", r"adidas", r"yeezy", r"air\s?max", r"dunk", r"boost",
    # 你可以换成药品/美食/steam 测评的关键词
]

MIN_CHAR = 3           # 最短字符数
MAX_CHAR = 2000        # 最长字符数
MIN_WORDS = 2          # 最少英文词数（中文可忽略）
# ==========================

URL_RE   = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
MENT_RE  = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
ONLY_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
CODEY_RE = re.compile(r"[{}[\]<>$^*_=~`|\\]{3,}")  # 疑似代码/命令密集符
EMOJI_LIKE_RE = re.compile(r"[\U00010000-\U0010ffff]+")  # 大部分表情/符号
REPEAT_CHAR_RE = re.compile(r"(.)\1{4,}")  # 5个以上重复字符

SPAM_HINTS = [
    r"promo\s?code", r"discount", r"free\s+\w+", r"dm\s+me", r"whatsapp", r"telegram",
    r"line\s?id", r"follow\s+me\s+for\s+link", r"subscribe\s+for", r"giveaway"
]
SPAM_RE = re.compile("|".join(SPAM_HINTS), re.IGNORECASE)

def normalize(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    t = URL_RE.sub("<URL>", t)
    t = EMAIL_RE.sub("<EMAIL>", t)
    t = MENT_RE.sub("<USER>", t)
    return t

def english_word_count(text: str) -> int:
    # 简单按空白切分统计英文词（中文场景该值会偏低）
    return len([w for w in text.split() if w.isalpha()])

def looks_like_only_noise(t: str) -> bool:
    # 仅标点/仅表情/重复字符等
    if not t:
        return True
    if ONLY_PUNCT_RE.match(t):
        return True
    if REPEAT_CHAR_RE.search(t):
        return True
    # 只有链接/@/话题 且几乎无其他内容
    t_wo_urls = URL_RE.sub("", t)
    t_wo_mentions = MENT_RE.sub("", t_wo_urls)
    t_wo_hashtags = HASHTAG_RE.sub("", t_wo_mentions).strip()
    if len(t_wo_hashtags) == 0:
        return True
    # 代码风/符号密集
    if CODEY_RE.search(t):
        # 同时长度很长/或非自然语言占比高时判为无效
        sym_ratio = len(re.findall(r"[^\w\s]", t)) / max(1, len(t))
        if sym_ratio > 0.4 or len(t) > 400:
            return True
    return False

def lang_ok(text: str) -> bool:
    if ENSURE_LANG is None:
        return True
    try:
        from langdetect import detect
        lang = detect(text)
        if ENSURE_LANG == "en":
            return lang == "en"
        if ENSURE_LANG == "zh":
            return lang in ("zh-cn", "zh-tw", "zh")
        return True
    except Exception:
        # 检测失败时保守放行
        return True

def domain_ok(text: str) -> bool:
    if not APPLY_DOMAIN_FILTER:
        return True
    return any(re.search(pat, text, re.IGNORECASE) for pat in DOMAIN_KEYWORDS)

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.read_csv(INPUT_CSV)

    # 只保留我们需要的列（若某些列不在，自动忽略）
    keep_cols = ["video_id","video_title","video_published_at","comment_id",
                 "published_at","like_count","text","char_len","word_count"]
    cols = [c for c in keep_cols if c in df.columns]
    df = df[cols].copy()

    # 预清洗：去首尾空白
    df["text"] = df["text"].astype(str).fillna("").str.strip()

    # 完全重复去重（文本、同视频）
    df = df.drop_duplicates(subset=["video_id","text"])

    # 近似重复：标准化后去重
    df["norm_for_dupe"] = (df["text"]
                           .str.lower()
                           .str.replace(r"[^\w\s]", "", regex=True)
                           .str.replace(r"\s+", " ", regex=True)
                           .str.strip())
    df = df.drop_duplicates(subset=["video_id","norm_for_dupe"])
    df = df.drop(columns=["norm_for_dupe"])

    # 规则过滤
    reasons = []
    clean_texts = []
    for t in df["text"].tolist():
        reason: Optional[str] = None
        nt = normalize(t)

        if len(nt) < MIN_CHAR:
            reason = "too_short"
        elif len(nt) > MAX_CHAR:
            reason = "too_long"
        elif looks_like_only_noise(nt):
            reason = "noise_only"
        elif SPAM_RE.search(nt):
            reason = "spam_like"
        elif not lang_ok(nt):
            reason = "lang_filter"
        elif not domain_ok(nt):
            reason = "domain_mismatch"
        else:
            # 英文词数阈值（仅当主要为英文文本时才严格）
            if english_word_count(nt) < MIN_WORDS and len(nt) < 8:
                reason = "too_short_words"

        reasons.append(reason)
        clean_texts.append(nt)

    df["clean_text"] = clean_texts
    df["dropped_reason"] = reasons

    # 保留通过的
    keep = df["dropped_reason"].isna()
    kept_df = df[keep].copy()

    # 更新长度统计（以 clean_text 为准）
    kept_df["char_len"] = kept_df["clean_text"].str.len()
    kept_df["word_count"] = kept_df["clean_text"].str.split().apply(len)

    kept_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved cleaned -> {OUTPUT_CSV}")
    print(f"Kept {len(kept_df)} / {len(df)} rows ({len(kept_df)/max(1,len(df)):.1%})")
    # 也可输出被删除样本，便于抽查
    dropped = df[~keep]
    if len(dropped) > 0:
        dropped.to_csv(OUTPUT_CSV.replace(".csv","_dropped.csv"), index=False)
        print(f"Dropped {len(dropped)} rows -> {OUTPUT_CSV.replace('.csv','_dropped.csv')}")

if __name__ == "__main__":
    main()
