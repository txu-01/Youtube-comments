import os, csv, time
from dotenv import load_dotenv
from googleapiclient.discovery import build

# 配置
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

SEARCH_QUERIES = [
    "medicine review",
    "drug review",
    "pharmaceutical review",
    "medication side effects",
    "supplement review"
]

MAX_VIDEOS = 10   # 每个搜索挑选若干视频
MAX_COMMENTS_PER_VIDEO = 500
OUT_CSV = "data/raw/pharma_comments.csv"

FIELDS = ["video_id","video_title","video_published_at",
          "comment_id","published_at","like_count","text","char_len","word_count"]

def search_videos(query, max_results=5):
    resp = youtube.search().list(
        q=query,
        part="id,snippet",
        maxResults=max_results,
        type="video",
        order="viewCount"
    ).execute()
    videos = []
    for it in resp.get("items", []):
        videos.append({
            "video_id": it["id"]["videoId"],
            "title": it["snippet"]["title"],
            "published_at": it["snippet"]["publishedAt"]
        })
    return videos

def fetch_comments(video_id, max_comments=500):
    comments = []
    token = None
    while True:
        resp = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=token,
            textFormat="plainText",
            order="time"
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
                "word_count": len(text.split())
            })
            if max_comments and len(comments) >= max_comments:
                return comments
        token = resp.get("nextPageToken")
        if not token:
            break
        time.sleep(0.1)
    return comments

def main():
    all_comments = []
    for query in SEARCH_QUERIES:
        videos = search_videos(query, max_results=MAX_VIDEOS//len(SEARCH_QUERIES))
        for v in videos:
            print(f"Fetching comments for: {v['title']}")
            rows = fetch_comments(v["video_id"], MAX_COMMENTS_PER_VIDEO)
            for r in rows:
                all_comments.append({
                    "video_id": v["video_id"],
                    "video_title": v["title"],
                    "video_published_at": v["published_at"],
                    **r
                })
            print(f"  Collected {len(rows)} comments")
    # 保存
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_comments)
    print(f"\nSaved {len(all_comments)} comments → {OUT_CSV}")

if __name__ == "__main__":
    main()
