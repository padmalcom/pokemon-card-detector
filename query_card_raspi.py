import sys
import signal
from picamera2 import Picamera2

from PyQt6.QtGui import QImage, QPainter, QColor, QBrush, QPixmap, QGuiApplication
from PyQt6.QtCore import QTimer, Qt, QRectF

from PyQt6.QtWidgets import (
  QApplication, QMainWindow, QGraphicsView,
  QGraphicsScene, QWidget, QVBoxLayout,
  QGraphicsPixmapItem
)

import cv2
import torch
import clip
import faiss
import numpy as np
from PIL import Image
import os
import json
from PIL import Image, ImageDraw, ImageFont

from huggingface_hub import from_pretrained_fastai

signal.signal(signal.SIGINT, signal.SIG_DFL)

class MainWindow(QMainWindow):
  def __init__(self):
    super().__init__()

    # init model
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    self.device = "mps" if torch.backends.mps.is_available() else "cpu"
    self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
    self.index = faiss.read_index("cards.faiss")
    self.labels = np.load("card_labels.npy")

    # init fake detector model
    self.fake_detector = from_pretrained_fastai('hugginglearners/pokemon-card-checker')

    # load cards
    with open("cards.json", "r", encoding="utf-8") as f:
      self.cards = json.load(f)

    if self.cards is None:
      raise("Could not read cards.")

    # init ui
    self.is_fit = False
    self.detect_card_now = False
    self.detect_fake_now = False
    self.detection_result = None

    self.scene = QGraphicsScene()

    self.pixmap_item = QGraphicsPixmapItem()
    self.scene.addItem(self.pixmap_item)

    screen = QGuiApplication.primaryScreen()
    geometry = screen.geometry()
    self.setGeometry(geometry)

    self.view = QGraphicsView(self.scene)
    self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
    self.view.setBackgroundBrush(QBrush(QColor("black")))
    self.view.setViewportUpdateMode(
      QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate
    )
    self.view.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    self.view.mousePressEvent = self.mouse_press_event

    self.setCentralWidget(self.view)
    self.picam2 = Picamera2()
    config = self.picam2.create_preview_configuration(
      main={"format": "RGB888", "size": (640, 480)}
    )
    self.picam2.configure(config)
    self.picam2.start()

    self.timer = QTimer()
    self.timer.timeout.connect(self.update_frame)
    self.timer.start(30)

  def mouse_press_event(self, event):
    print("Event:", event)
    if self.detection_result is not None:
      self.detection_result = None
      self.detect_card_now = False
      self.detect_fake_now = False
    else:
      if event.localPos.x > 400:
        self.detect_fake_now = True
      else:
        self.detect_card_now = True

  def update_frame(self):
    frame = self.picam2.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    h, w, ch = frame.shape
    bytes_per_line = ch * w

    #qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    if self.detect_card_now == True:
      print("detecting...")
      emb = self.embed_frame(frame)
      dist, idx = self.index.search(emb, 1)
      print("Dist:", dist[0][0])
      if dist[0][0] < 0.4:
        card_name = self.labels[idx[0][0]][:-4]
        print("Detected card:", card_name)
        if card_name in self.cards:
          card = self.cards[card_name]
          card_name = card["name"]
          card_rarity = card["rarity"]
          card_price = card["price"]
          self.detection_result = [f"{card_name} ({dist[0][0]:.4f})", f"Seltenheit: {card_rarity}", f"Wert: {card_price}€"]
      else:
        pass
        self.detection_result = ["Nichts gefunden"]
      self.detect_now = False

    if self.detect_fake_now == True:
      pred_label, _, scores = self.fake_detector.predict(frame)
      scores = scores.detach().numpy()
      scores = {'real': float(scores[1]), 'fake': float(scores[0])}

      if scores[1] > scores[0]:
        self.detection_result = ["Echt!"]
      else:
        self.detection_result = ["Fake!"]

    if self.detection_result is not None:
      print("Result is: ", self.detection_result)
      for idx, s in enumerate(self.detection_result):
        frame = self.draw_on_frame(frame, s, idx)

    qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)
    self.pixmap_item.setPixmap(pixmap)

    if self.is_fit == False:
      self.fit_pixmap()
      self.is_fit = True

  def fit_pixmap(self):
    if not self.pixmap_item.pixmap().isNull():
      self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

  def embed_frame(self, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    img_input = self.preprocess(pil).unsqueeze(0).to(self.device)
    with torch.no_grad():
      emb = self.model.encode_image(img_input)
      emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")

  def draw_on_frame(self, frame, text, line=0):
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype("Times New Roman.ttf", 40)
    draw.text((50, (line+1)*50), text, font=font, fill=(0, 0, 0))
    return np.array(pil_img)

if __name__ == "__main__":
  app = QApplication(sys.argv)
  window = MainWindow()
  window.showFullScreen()
  sys.exit(app.exec())
