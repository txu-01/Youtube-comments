import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
from tqdm import tqdm

# ========= 配置 =========
INPUT_CSV  = "data/processed/all_domains_merged.csv"
OUTPUT_CSV = "data/processed/all_domains_with_sentiment.csv"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =======================

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

print("Loading data...")
df = pd.read_csv(INPUT_CSV)
texts = df["clean_text"].fillna("").tolist()

sentiments = []
probs_pos, probs_neu, probs_neg = [], [], []

print(f"Running inference on {len(texts)} comments...")
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch_texts = texts[i:i+BATCH_SIZE]
    tokens = tokenizer(batch_texts, padding=True, truncation=True,
                       max_length=128, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**tokens)
        scores = softmax(outputs.logits, dim=1).cpu().numpy()

    for s in scores:
        neg, neu, pos = s
        label = "positive" if pos == max(s) else ("negative" if neg == max(s) else "neutral")
        sentiments.append(label)
        probs_pos.append(round(float(pos),4))
        probs_neu.append(round(float(neu),4))
        probs_neg.append(round(float(neg),4))

df["sentiment"] = sentiments
df["prob_positive"] = probs_pos
df["prob_neutral"]  = probs_neu
df["prob_negative"] = probs_neg

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved → {OUTPUT_CSV}")
