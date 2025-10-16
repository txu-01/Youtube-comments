import os, csv, time, datetime as dt
from typing import List, Dict, Set
from dotenv import load_dotenv
from googleapiclient.discovery import build

# ============== 配置 ==============
SEARCH_QUERIES = [
    "steam review",
    "pc game review",
    "steam game review",
    "game review 2025",
    "pc game impressions",
    "steam early access review"
]
TOP_VIDEOS_PER_QUERY = 5          # 每个关键词拿前 N 个热门视频
PUBLISHED_AFTER = "2025-01-01T00:00:00Z"  # 仅收集该日期之后发布的视频；设为 None 则不限制
MAX_COMMENTS_PER_VIDEO = 400      # 每个视频最多抓多少条评论；None 表示抓到尽
ORDER = "viewCount"               # "viewCount" 或 "relevance" 或 "date"
REGION_CODE = None                # 如需要按地区，可设 "US" / "GB" / "JP" / "CN" 等，否则 None
LANG_FILTER = None                # 仅用于搜索时的倾向提示，不是硬性过滤；可用 "en" 等
OUT_CSV = "data/raw/steam_review_comments_2025.csv"
SLEEP_BETWEEN_PAGES = 0.1
# =================================

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def init_youtube():
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing YOUTUBE_API_KEY in .env")
    return build("youtube", "v3", developerKey=api_key)

def search_top_videos(youtube, query: str, top_n: int) -> List[Dict]:
    """搜索某关键词的热门视频，返回视频基本信息列表"""
    params = dict(
        q=query,
        part="id,snippet",
        maxResults=50,        # 先拿最多 50，再从中选前 top_n
        type="video",
        order=ORDER
    )
    if PUBLISHED_AFTER:
        params["publishedAfter"] = PUBLISHED_AFTER
    if REGION_CODE:
        params["regionCode"] = REGION_CODE
    if LANG_FILTER:
        params["relevanceLanguage"] = LANG_FILTER

    resp = youtube.search().list(**params).execute()
    items = resp.get("items", [])
    video_ids = [it["id"]["videoId"] for it in items]
    video_ids = video_ids[:top_n]  # 选前 top_n

    # 拉取更完整的元数据（标题/发布时间/观看数）
    videos = []
    if video_ids:
        detail = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(video_ids)
        ).execute()
        for it in detail.get("items", []):
            videos.append({
                "video_id": it["id"],
                "title": it["snippet"]["title"],
                "published_at": it["snippet"]["publishedAt"],
                "view_count": int(it["statistics"].get("viewCount", 0))
            })
    # 再按观看数排序，保证热门
    videos.sort(key=lambda x: x["view_count"], reverse=True)
    return videos

def fetch_top_level_comments(youtube, video_id: str, max_comments=None) -> List[Dict]:
    """抓取顶层评论（不展开回复）"""
    comments = []
    token = None
    while True:
        resp = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=token,
            textFormat="plainText",
            order="time"  # 也可 "relevance"
        ).execute()

        for it in resp.get("items", []):
            snip = it["snippet"]["topLevelComment"]["snippet"]
            text = (snip.get("textDisplay") or "").replace("\n", " ").strip()
            comments.append({
                "comment_id": it["snippet"]["topLevelComment"]["id"],
                "published_at": snip.get("publishedAt"),
                "like_count": snip.get("likeCount", 0),
                "text": text,
                "char_len": len(text),
                "word_count": len([w for w in text.split() if w])
            })
            if max_comments and len(comments) >= max_comments:
                return comments

        token = resp.get("nextPageToken")
        if not token:
            break
        time.sleep(SLEEP_BETWEEN_PAGES)
    return comments

def main():
    yt = init_youtube()
    ensure_dir(OUT_CSV)

    # 1) 搜索各关键词的热门视频并去重
    print("Searching top videos for Steam/PC game reviews...")
    video_map: Dict[str, Dict] = {}
    for q in SEARCH_QUERIES:
        vids = search_top_videos(yt, q, TOP_VIDEOS_PER_QUERY)
        for v in vids:
            # 按 video_id 去重，不同关键词命中的相同视频只保留一次
            video_map.setdefault(v["video_id"], v)

    videos = list(video_map.values())
    videos.sort(key=lambda x: x["view_count"], reverse=True)
    print(f"Selected {len(videos)} unique videos.")
    for v in videos:
        print(f"- {v['video_id']} | {v['view_count']:>10,d} views | {v['title']}")

    # 2) 抓评论并保存
    fields = ["video_id","video_title","video_published_at",
              "comment_id","published_at","like_count","text","char_len","word_count"]
    total = 0
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for v in videos:
            print(f"\nFetching comments for: {v['title']}")
            rows = fetch_top_level_comments(yt, v["video_id"], MAX_COMMENTS_PER_VIDEO)
            for r in rows:
                w.writerow({
                    "video_id": v["video_id"],
                    "video_title": v["title"],
                    "video_published_at": v["published_at"],
                    **r
                })
            total += len(rows)
            print(f"  collected {len(rows)} comments")

    print(f"\nSaved {total} comments → {OUT_CSV}")

if __name__ == "__main__":
    main()
