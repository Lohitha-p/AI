import os
import torch
import clip
from PIL import Image
import pandas as pd
import numpy as np

# =====================
# Config
# =====================
DATA_CSV = "styles.csv"
IMAGE_FOLDER = "images"
EMBEDDINGS_FILE = "clip_image_embeddings.npy"
MAX_IMAGES = 20000  # Limit to first 20,000 images

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# Load dataset
# =====================
df = pd.read_csv(DATA_CSV, quotechar='"', on_bad_lines='skip')
df['image_path'] = df['id'].astype(str) + ".jpg"
df['image_path'] = df['image_path'].apply(lambda x: os.path.join(IMAGE_FOLDER, x))

# Limit to first MAX_IMAGES
df = df.head(MAX_IMAGES)
print(f"Processing {len(df)} images...")

# =====================
# Load CLIP model
# =====================
model, preprocess = clip.load("ViT-B/32", device=device)

# =====================
# Generate embeddings
# =====================
embeddings = []

for idx, row in df.iterrows():
    if not os.path.exists(row['image_path']):
        print(f"Image not found: {row['image_path']}")
        emb = np.zeros(512, dtype='float32')  # CLIP embedding dimension
    else:
        img = preprocess(Image.open(row['image_path']).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            emb_tensor = model.encode_image(img)
        emb = emb_tensor.cpu().numpy().flatten()
    embeddings.append(emb)
    if (idx + 1) % 50 == 0:
        print(f"Processed {idx + 1}/{len(df)} images")

embeddings = np.array(embeddings).astype('float32')
np.save(EMBEDDINGS_FILE, embeddings)
print(f"Saved embeddings for {len(df)} images to {EMBEDDINGS_FILE}")
