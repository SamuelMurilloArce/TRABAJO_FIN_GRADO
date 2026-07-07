import cv2
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt


class VideoLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(720, 420)
        self.setStyleSheet(
            """
            QLabel {
                background-color: #0f172a;
                border-radius: 10px;
                color: #e2e8f0;
                font-size: 14pt;
            }
            """
        )
        self.setText("Carga un vídeo para comenzar")

    def set_frame(self, frame_bgr):
        if frame_bgr is None:
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width

        image = QImage(
            frame_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(
            self.width(),
            self.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.setPixmap(scaled_pixmap)
