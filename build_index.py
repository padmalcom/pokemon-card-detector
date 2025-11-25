import os
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm

CARD_FOLDER = "data"
OUTPUT_EMBEDDINGS = "card_embeddings.npy"
OUTPUT_LABELS = "card_labels.npy"

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=device)

embeddings = []
labels = []

print("Encoding card images...")
for filename in tqdm(os.listdir(CARD_FOLDER)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(CARD_FOLDER, filename)

    img = Image.open(path).convert("RGB")
    img_input = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(img_input)
        emb = emb / emb.norm(dim=-1, keepdim=True)   # normalize

    embeddings.append(emb.cpu().numpy())
    labels.append(filename)

embeddings = np.concatenate(embeddings, axis=0)
labels = np.array(labels)

print("Saving embeddings...")
np.save(OUTPUT_EMBEDDINGS, embeddings)
np.save(OUTPUT_LABELS, labels)

print("Done! Encoded", len(labels), "cards.")
