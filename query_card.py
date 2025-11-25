import torch
import clip
import faiss
import numpy as np
from PIL import Image
import os
import json

os.environ['KMP_DUPLICATE_LIB_OK']='True'


device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load CLIP
model, preprocess = clip.load("ViT-B/32", device=device)

# Load FAISS index + labels
index = faiss.read_index("cards.faiss")
labels = np.load("card_labels.npy")

def encode_image(path):
    img = Image.open(path).convert("RGB")
    img_input = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_input)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")

def search_card(image_path, k=1):
    query = encode_image(image_path)
    distances, ids = index.search(query, k)
    return distances, ids

# Example usage:
test_image = "pokemon_card.webp"
dist, idx = search_card(test_image)

card_id = labels[idx[0][0]]
print("Best match:", card_id)
print("Distance:", dist[0][0])

with open("cards.json", "r", encoding="utf-8") as f:
    cards = json.load(f)
    print(cards[card_id[:-4]])
