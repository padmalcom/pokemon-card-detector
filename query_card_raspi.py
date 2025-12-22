import sys
import signal
from picamera2 import Picamera2

from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer

from PyQt6.QtWidgets import (
  QApplication, QMainWindow, QGraphicsView,
  QGraphicsScene, QWidget, QVBoxLayout,
  QGraphicsPixmapItem
)

signal.signal(signal.SIGINT, signal.SIG_DFL)

class MainWindow(QMainWindow):
  def __init__(self):
    super().__init__()

    scene = QGraphicsScene()
    #scene.addText("Hello QGraphicsView!")

    self.pixmap_item = QGraphicsPixmapItem()
    scene.addItem(self.pixmap_item)

    view = QGraphicsView(scene)

    central_widget = QWidget()
    layout = QVBoxLayout(central_widget)
    layout.addWidget(view)

    self.setCentralWidget(central_widget)

    self.picam2 = Picamera2()
    config = self.picam2.create_preview_configuration(
      main={"format": "RGB888", "size": (640, 480)}
    )
    self.picam2.configure(config)
    self.picam2.start()

    self.timer = QTimer()
    self.timer.timeout.connect(self.update_frame)
    self.timer.start(30)  # ~33 FPS

  def update_frame(self):
    frame = self.picam2.capture_array()

    h, w, ch = frame.shape
    bytes_per_line = ch * w

    qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

    pixmap = QPixmap.fromImage(qimg)
    self.pixmap_item.setPixmap(pixmap)

    # Optional: auto-fit view
    #self.fitInView(self.pixmap_item, mode=1)

if __name__ == "__main__":
  app = QApplication(sys.argv)
  window = MainWindow()
  window.showFullScreen()
  sys.exit(app.exec())
