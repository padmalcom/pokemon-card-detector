import cv2
import torch
import clip
import faiss
import numpy as np
from PIL import Image
import os
import json
from PIL import Image, ImageDraw, ImageFont

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = "mps" if torch.backends.mps.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

index = faiss.read_index("cards.faiss")
labels = np.load("card_labels.npy")

def embed_frame(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    img_input = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_input)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")

def draw_on_frame(frame, text, line=0):
    pil_img = Image.fromarray(frame)

    # Prepare to draw text
    draw = ImageDraw.Draw(pil_img)
    # Use a TTF font that supports the character
    font = ImageFont.truetype("Times New Roman.ttf", 40)  # adjust font path and size accordingly
    #font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)

    # Draw text with UTF-8 character
    draw.text((50, (line+1)*50), text, font=font, fill=(0, 0, 0))  # black color

    # Convert back to OpenCV image
    return np.array(pil_img)

ALPHA = 0.6

cap = cv2.VideoCapture(0)

# calculate the center area of the screen
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cut_out_height = frame_height - 60
cut_out_width = cut_out_height * (63 / 88)

center_x = frame_width // 2
center_y = frame_height // 2

top_left_x = int(center_x - cut_out_width // 2)
top_left_y = int(center_y - cut_out_height // 2)
bottom_right_x = int(center_x + cut_out_width // 2)
bottom_right_y = int(center_y + cut_out_height // 2)

with open("cards.json", "r", encoding="utf-8") as f:
    cards = json.load(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        grey_overlay = np.full(frame.shape, (128, 128, 128), dtype=np.uint8)

        blended = cv2.addWeighted(frame, 1 - ALPHA, grey_overlay, ALPHA, 0)

        # Cut out the rectangle area
        blended[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        cut_out = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        emb = embed_frame(cut_out)
        dist, idx = index.search(emb, 1)
        if dist[0][0] < 0.3:
            card_name = labels[idx[0][0]][:-4]
            if card_name in cards:
                card = cards[card_name]
                card_name = card["name"]
                card_rarity = card["rarity"]
                card_price = card["price"]
                blended = draw_on_frame(blended, f"{card_name} ({dist[0][0]:.4f})", 0)
                blended = draw_on_frame(blended, f"Seltenheit: {card_rarity}", 1)
                blended = draw_on_frame(blended, f"Wert: {card_price}€", 2)
        cv2.imshow("Scanner", blended)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
