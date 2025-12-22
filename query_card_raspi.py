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

signal.signal(signal.SIGINT, signal.SIG_DFL)

class MainWindow(QMainWindow):
  def __init__(self):
    super().__init__()

    # init model
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    self.device = "mps" if torch.backends.mps.is_available() else "cpu"
    self.model, self.preprocess = clip.load("ViT-B/32", device=device)
    self.index = faiss.read_index("cards.faiss")
    self.labels = np.load("card_labels.npy")

    # load cards
    with open("cards.json", "r", encoding="utf-8") as f:
      cards = json.load(f)
    if cards is None:
      raise("Could not read cards.")

    # init ui
    self.is_fit = False
    self.detect_now = False

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
    self.detect_now = not self.detect_now

  def update_frame(self):
    frame = self.picam2.capture_array()

    h, w, ch = frame.shape
    bytes_per_line = ch * w

    qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

    pixmap = QPixmap.fromImage(qimg)
    self.pixmap_item.setPixmap(pixmap)

    if self.is_fit == False:
      self.fit_pixmap()
      self.is_fit = True

    if self.detect_now == True:
      emb = self.embed_frame(frame)

  def fit_pixmap(self):
    if not self.pixmap_item.pixmap().isNull():
      self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

  def embed_frame(self, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    img_input = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
      emb = model.encode_image(img_input)
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
