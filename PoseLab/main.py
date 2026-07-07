import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QFile

from app.main_window import MainWindow


def load_stylesheet(app: QApplication) -> None:
    style_path = Path("app/styles/style.qss")
    if style_path.exists():
        file = QFile(str(style_path))
        if file.open(QFile.ReadOnly | QFile.Text):
            app.setStyleSheet(str(file.readAll(), encoding="utf-8"))


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("PoseLab")

    load_stylesheet(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
