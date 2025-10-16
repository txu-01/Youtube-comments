# -*- coding: utf-8 -*-
"""
按“本年”发布的热门视频（viewCount）抓取评论到 data2/raw/
- 每个视频最多 500 条（不足则拿多少算多少）
- 直到该领域累计达到 TARGET_TOTAL（默认 20000）
依赖: google-api-python-client python-dotenv pandas tqdm
.env: YOUTUBE_API_KEY=xxxx
用法示例:
    python fetch_comments_v2.py --domain sneaker --target 20000
    python fetch_comments_v2.py --domain pharma  --target 20000
    python fetch_comments_v2.py --domain food    --target 20000
    python fetch_comments_v2.py --domain steam   --target 20000
"""

import os, time, argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from settings import RAW_DIR  # 确保 settings.py 已设置 BASE_DATA_DIR=data2

# --------- 参数与查询词预设 ---------
PRESET_QUERIES = {
    "sneaker": [
        "sneaker review", "running shoes review", "basketball shoes review",
        "Nike shoes review", "Adidas shoes review", "barefoot shoes review",
        "Air Jordan review", "Asics running review", "Brooks running review"
    ],
    "pharma": [
        "medicine review", "drug review", "supplement review",
        "over the counter medicine review", "pain relief review",
        "asthma inhaler review", "antibiotics review"
    ],
    "food": [
        "food review", "restaurant review", "fast food review",
        "menu review", "taste test review", "snack review"
    ],
    "steam": [
        "steam game review", "pc game review 2025", "new game review",
        "indie game review", "AAA game review", "game performance review"
    ],
}

def this_year_utc_window():
    now = datetime.now(timezone.utc)
    year_start = datetime(year=now.year, month=1, day=1, tzinfo=timezone.utc)
    next_year_start = datetime(year=now.year + 1, month=1, day=1, tzinfo=timezone.utc)
    # RFC3339 / ISO8601 (YouTube API 需要 'Z' 结尾)
    after = year_start.isoformat().replace("+00:00", "Z")
    before = next_year_start.isoformat().replace("+00:00", "Z")
    return after, before
# -----------------------------------

def build_youtube_client():
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing YOUTUBE_API_KEY in .env")
    return build("youtube", "v3", developerKey=api_key)

def search_videos_this_year(yt, query, pages=10, order="viewCount"):
    vids = []
    page_token = None
    published_after, published_before = this_year_utc_window()
    for _ in range(pages):
        resp = yt.search().list(
            q=query,
            part="id",
            type="video",
            maxResults=50,
            order=order,
            publishedAfter=published_after,
            publishedBefore=published_before,
            pageToken=page_token
        ).execute()
        for item in resp.get("items", []):
            vids.append(item["id"]["videoId"])
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return vids

def fetch_comments_for_video(yt, video_id, per_video_limit=500, order_comments_by="relevance"):
    """抓取单视频顶层评论（不展开回复），最多 per_video_limit 条。"""
    rows, grabbed = [], 0
    page_token = None
    try:
        vi = yt.videos().list(part="snippet", id=video_id).execute()
        if not vi["items"]:
            return rows
        snippet = vi["items"][0]["snippet"]
        video_title = snippet.get("title", "")
        video_published_at = snippet.get("publishedAt", "")

        while True:
            resp = yt.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                textFormat="plainText",
                order=order_comments_by,   # "relevance" or "time"
                pageToken=page_token
            ).execute()

            items = resp.get("items", [])
            if not items and not page_token:
                break  # 无评论或被关闭

            for it in items:
                top = it["snippet"]["topLevelComment"]["snippet"]
                rows.append({
                    "video_id": video_id,
                    "video_title": video_title,
                    "video_published_at": video_published_at,
                    "comment_id": it["snippet"]["topLevelComment"]["id"],
                    "published_at": top.get("publishedAt", ""),
                    "like_count": top.get("likeCount", 0),
                    "text": top.get("textDisplay", ""),
                    "domain": None,  # 调用处填
                })
                grabbed += 1
                if grabbed >= per_video_limit:
                    return rows

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    except HttpError as e:
        # 常见: commentsDisabled / quotaExceeded
        print(f"[skip] video {video_id} - {e}")
    except Exception as e:
        print(f"[skip] video {video_id} - {e}")

    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, required=True,
                        choices=list(PRESET_QUERIES.keys()),
                        help="选择领域：sneaker / pharma / food / steam")
    parser.add_argument("--target", type=int, default=20000, help="目标总评论数")
    parser.add_argument("--pages", type=int, default=10, help="每个搜索词翻页数(每页50视频)")
    parser.add_argument("--per_video_limit", type=int, default=500, help="每视频评论上限")
    parser.add_argument("--order_comments_by", type=str, default="relevance",
                        choices=["relevance","time"], help="评论排序")
    parser.add_argument("--sleep", type=float, default=0.2, help="每个视频之间 sleep 秒数")
    args = parser.parse_args()

    domain = args.domain
    target_total = args.target
    per_video_limit = args.per_video_limit
    order_comments_by = args.order_comments_by
    sleep_between_videos = args.sleep
    queries = PRESET_QUERIES[domain]

    yt = build_youtube_client()
    out_file = RAW_DIR / f"{domain}_comments_raw.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1) 汇总“本年热门”视频列表
    video_ids = []
    for q in queries:
        video_ids.extend(
            search_videos_this_year(yt, q, pages=args.pages, order="viewCount")
        )
    # 去重并保持顺序
    seen = set(); uniq_ids = []
    for vid in video_ids:
        if vid not in seen:
            uniq_ids.append(vid); seen.add(vid)
    print(f"[{domain}] Candidate videos (this year): {len(uniq_ids)}")

    # 2) 逐视频抓取评论
    all_rows = []
    for vid in tqdm(uniq_ids, desc=f"Fetching {domain} comments"):
        rows = fetch_comments_for_video(
            yt, vid, per_video_limit=per_video_limit, order_comments_by=order_comments_by
        )
        # 填充 domain 字段
        for r in rows:
            r["domain"] = domain

        if rows:
            all_rows.extend(rows)

        # 断点保存（每 ~5000 条）
        if len(all_rows) >= 5000 and len(all_rows) % 5000 < 1000:
            pd.DataFrame(all_rows).drop_duplicates(
                subset=["video_id","comment_id"]
            ).to_csv(out_file, index=False)

        if len(all_rows) >= target_total:
            break

        time.sleep(sleep_between_videos)

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["video_id","comment_id"])
    df.to_csv(out_file, index=False)
    print(f"[{domain}] Saved {len(df)} rows → {out_file}")

if __name__ == "__main__":
    main()
