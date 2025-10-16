# settings.py
from pathlib import Path

# === 切换版本：只改这里就能在 v1 / v2 之间切换 ===
BASE_DATA_DIR = Path("data2")          # 现在使用 data2（v2）
FIG_DIR       = Path("figures_v2")     # 新版图表目录

# 子目录
RAW_DIR       = BASE_DATA_DIR / "raw"
PROCESSED_DIR = BASE_DATA_DIR / "processed"
RESULTS_DIR   = BASE_DATA_DIR / "results"

# 常用文件名（按你的现有列名/流程命名）
MERGED_CSV    = PROCESSED_DIR / "all_domains_merged.csv"
WITH_SENT_CSV = PROCESSED_DIR / "all_domains_with_sentiment.csv"

# 确保目录存在
for p in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)
