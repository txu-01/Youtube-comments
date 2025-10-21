# bert_sentiment_inference_v2.py
# Run sentiment inference with RoBERTa and save probabilities/labels.

from settings import MERGED_CSV, WITH_SENT_CSV
import pandas as pd
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 32
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

df = pd.read_csv(MERGED_CSV)
texts = df["clean_text"].fillna("").astype(str).tolist()

sentiments, ppos, pneu, pneg = [], [], [], []
for i in range(0, len(texts), BATCH_SIZE):
    batch = tok(
        texts[i:i+BATCH_SIZE],
        padding=True, truncation=True, max_length=MAX_LEN,
        return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**batch).logits
        probs = softmax(logits, dim=1).cpu().numpy()
    for s in probs:
        neg, neu, pos = s  # model order: [negative, neutral, positive]
        if pos >= max(s):
            label = "positive"
        elif neg >= max(s):
            label = "negative"
        else:
            label = "neutral"
        sentiments.append(label)
        ppos.append(float(pos)); pneu.append(float(neu)); pneg.append(float(neg))

df["sentiment"]     = sentiments
df["prob_positive"] = ppos
df["prob_neutral"]  = pneu
df["prob_negative"] = pneg

df.to_csv(WITH_SENT_CSV, index=False)
print(f"Saved → {WITH_SENT_CSV} ({len(df)} rows)")
