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

signal.signal(signal.SIGINT, signal.SIG_DFL)

class MainWindow(QMainWindow):
  def __init__(self):
    super().__init__()
    self.is_fit = False

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

  def fit_pixmap(self):
    if not self.pixmap_item.pixmap().isNull():
      self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

if __name__ == "__main__":
  app = QApplication(sys.argv)
  window = MainWindow()
  window.showFullScreen()
  sys.exit(app.exec())
