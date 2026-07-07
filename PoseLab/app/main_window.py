
from pathlib import Path

from PySide6.QtCore import QTimer, Qt

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QGroupBox,
    QGridLayout,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QProgressBar,
)

from app.widgets.video_label import VideoLabel
from core.video_manager import VideoManager
from core.metrics import MetricsCalculator
from core.anomaly_detector import AnomalyDetector
from core.exporters import export_keypoints_csv, export_anomalies_csv

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PoseLab - Análisis de detección de pose")
        self.resize(1280, 820)

        self.video_manager = VideoManager()
        self.detector = None
        self.metrics = MetricsCalculator()
        self.anomaly_detector = AnomalyDetector()

        self.frame_results = []
        self.anomalies = []

        self.current_video_path = None
        self.processing = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_next_frame)

        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)

        title = QLabel("PoseLab")
        title.setObjectName("TitleLabel")

        subtitle = QLabel(
            "Herramienta de análisis técnico de modelos de detección de pose en vídeos 2D"
        )
        subtitle.setObjectName("SubtitleLabel")

        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        content_layout = QHBoxLayout()

        sidebar = self.create_sidebar()
        video_area = self.create_video_area()

        content_layout.addWidget(sidebar, 0)
        content_layout.addWidget(video_area, 1)

        main_layout.addLayout(content_layout)

        self.setCentralWidget(central)

    def create_sidebar(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        panel.setFixedWidth(330)

        file_group = QGroupBox("Vídeo")
        file_layout = QVBoxLayout(file_group)

        self.video_path_label = QLabel("Ningún vídeo cargado")
        self.video_path_label.setWordWrap(True)

        self.load_button = QPushButton("Cargar vídeo")
        self.load_button.clicked.connect(self.load_video)

        file_layout.addWidget(self.video_path_label)
        file_layout.addWidget(self.load_button)

        model_group = QGroupBox("Modelo")
        model_layout = QVBoxLayout(model_group)

        self.model_combo = QComboBox()
        self.model_combo.addItem("YOLOv8n-pose")

        model_layout.addWidget(QLabel("Modelo de detección:"))
        model_layout.addWidget(self.model_combo)

        process_group = QGroupBox("Procesamiento")
        process_layout = QVBoxLayout(process_group)

        self.process_button = QPushButton("Procesar vídeo")
        self.process_button.clicked.connect(self.toggle_processing)
        self.process_button.setEnabled(False)

        self.reset_button = QPushButton("Reiniciar")
        self.reset_button.clicked.connect(self.reset_processing)
        self.reset_button.setEnabled(False)

        self.progress = QProgressBar()
        self.progress.setValue(0)

        process_layout.addWidget(self.process_button)
        process_layout.addWidget(self.reset_button)
        process_layout.addWidget(self.progress)

        export_group = QGroupBox("Exportación")
        export_layout = QVBoxLayout(export_group)

        self.export_keypoints_button = QPushButton("Exportar keypoints CSV")
        self.export_keypoints_button.clicked.connect(self.export_keypoints)
        self.export_keypoints_button.setEnabled(False)

        self.export_anomalies_button = QPushButton("Exportar anomalías CSV")
        self.export_anomalies_button.clicked.connect(self.export_anomalies)
        self.export_anomalies_button.setEnabled(False)

        export_layout.addWidget(self.export_keypoints_button)
        export_layout.addWidget(self.export_anomalies_button)

        layout.addWidget(file_group)
        layout.addWidget(model_group)
        layout.addWidget(process_group)
        layout.addWidget(export_group)
        layout.addStretch()

        return panel

    def create_video_area(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.video_label = VideoLabel()

        metrics_group = QGroupBox("Métricas generales")
        metrics_layout = QGridLayout(metrics_group)

        self.metric_total_frames = self.create_metric_label("0")
        self.metric_valid_frames = self.create_metric_label("0")
        self.metric_confidence = self.create_metric_label("0.00")
        self.metric_fps = self.create_metric_label("0.00")
        self.metric_anomalies = self.create_metric_label("0")

        metrics_layout.addWidget(QLabel("Frames procesados"), 0, 0)
        metrics_layout.addWidget(self.metric_total_frames, 1, 0)

        metrics_layout.addWidget(QLabel("Frames válidos"), 0, 1)
        metrics_layout.addWidget(self.metric_valid_frames, 1, 1)

        metrics_layout.addWidget(QLabel("Confianza media"), 0, 2)
        metrics_layout.addWidget(self.metric_confidence, 1, 2)

        metrics_layout.addWidget(QLabel("FPS estimados"), 0, 3)
        metrics_layout.addWidget(self.metric_fps, 1, 3)

        metrics_layout.addWidget(QLabel("Anomalías"), 0, 4)
        metrics_layout.addWidget(self.metric_anomalies, 1, 4)

        anomalies_group = QGroupBox("Últimas anomalías detectadas")
        anomalies_layout = QVBoxLayout(anomalies_group)

        self.anomalies_table = QTableWidget(0, 5)
        self.anomalies_table.setHorizontalHeaderLabels(
            ["Frame", "Tiempo", "Tipo", "Keypoint", "Valor"]
        )
        self.anomalies_table.horizontalHeader().setStretchLastSection(True)

        anomalies_layout.addWidget(self.anomalies_table)

        layout.addWidget(self.video_label, 4)
        layout.addWidget(metrics_group, 0)
        layout.addWidget(anomalies_group, 2)

        return panel

    def create_metric_label(self, text):
        label = QLabel(text)
        label.setObjectName("MetricValue")
        label.setAlignment(Qt.AlignCenter)
        return label

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar vídeo",
            "data/input_videos",
            "Vídeos (*.mp4 *.avi *.mov *.mkv);;Todos los archivos (*.*)",
        )

        if not file_path:
            return

        try:
            self.video_manager.open(file_path)
        except RuntimeError as error:
            QMessageBox.critical(self, "Error", str(error))
            return

        self.current_video_path = file_path
        print("current_video_path =", self.current_video_path)
        self.video_path_label.setText(Path(file_path).name)

        ok, frame = self.video_manager.read()
        if ok:
            self.video_label.set_frame(frame)

        self.video_manager.reset()

        self.reset_processing(clear_video=False)

        self.current_video_path = file_path
        self.process_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.process_button.setText("Procesar vídeo")

        
    def create_detector(self):
        model_name = self.model_combo.currentText()

        if model_name == "YOLOv8n-pose":
            from models.yolov8_pose_detector import YOLOv8PoseDetector
            return YOLOv8PoseDetector()

        raise ValueError(f"Modelo no soportado: {model_name}")

    def toggle_processing(self):
        if self.processing:
            self.timer.stop()
            self.processing = False
            self.process_button.setText("Continuar procesamiento")
            return
        
        QMessageBox.information(self, "Debug", f"Botón Procesar pulsado\ncurrent_video_path = {self.current_video_path}")

        if not self.current_video_path:
            QMessageBox.warning(self, "Error", "No hay vídeo cargado.")
            return

        try:
            if self.detector is None:
                print("Creando detector...")
                self.detector = self.create_detector()
                print("Detector creado correctamente")

            self.processing = True
            self.process_button.setText("Pausar")
            self.timer.start(30)
        
        except Exception as error:
            import traceback
            traceback.print_exc()

            QMessageBox.critical(
                self,
                "Error creando detector",
                str(error)
            )

    def process_next_frame(self):
        print("process_next_frame llamado")

        try:
            ok, frame = self.video_manager.read()
            print("read:", ok, "frame:", frame is not None)

            if not ok:
                self.finish_processing()
                return

            frame_id = self.video_manager.current_frame_index()
            timestamp = self.video_manager.timestamp_for_frame(frame_id)

            print("detectando frame", frame_id)

            result, raw_result = self.detector.detect(frame, frame_id, timestamp)

            print("dibujando frame", frame_id)

            output_frame = self.detector.draw(frame, raw_result)

            frame_anomalies = self.anomaly_detector.analyze_frame(result)

            self.frame_results.append(result)
            self.anomalies.extend(frame_anomalies)
            self.metrics.add_result(result)

            self.video_label.set_frame(output_frame)
            self.update_metrics()
            self.update_anomalies_table(frame_anomalies)
            self.update_progress()

        except Exception as error:
            self.timer.stop()
            self.processing = False
            self.process_button.setText("Procesar vídeo")

            import traceback
            traceback.print_exc()

            QMessageBox.critical(
                self,
                "Error durante el procesamiento",
                str(error),
        )

    def update_progress(self):
        if self.video_manager.total_frames <= 0:
            self.progress.setValue(0)
            return

        current = self.video_manager.current_frame_index()
        percent = int((current / self.video_manager.total_frames) * 100)
        self.progress.setValue(min(percent, 100))

    def update_metrics(self):
        summary = self.metrics.summary()

        self.metric_total_frames.setText(str(summary["total_frames"]))
        self.metric_valid_frames.setText(str(summary["valid_frames"]))
        self.metric_confidence.setText(f'{summary["mean_confidence"]:.2f}')
        self.metric_fps.setText(f'{summary["estimated_fps"]:.2f}')
        self.metric_anomalies.setText(str(len(self.anomalies)))

    def update_anomalies_table(self, new_anomalies):
        for anomaly in new_anomalies:
            row = self.anomalies_table.rowCount()
            self.anomalies_table.insertRow(row)

            self.anomalies_table.setItem(row, 0, QTableWidgetItem(str(anomaly["frame_id"])))
            self.anomalies_table.setItem(row, 1, QTableWidgetItem(f'{anomaly["timestamp"]:.2f}'))
            self.anomalies_table.setItem(row, 2, QTableWidgetItem(anomaly["type"]))
            self.anomalies_table.setItem(row, 3, QTableWidgetItem(anomaly["keypoint"]))
            self.anomalies_table.setItem(row, 4, QTableWidgetItem(str(anomaly["value"])))

        if self.anomalies_table.rowCount() > 200:
            self.anomalies_table.removeRow(0)

    def finish_processing(self):
        self.timer.stop()
        self.processing = False
        self.process_button.setText("Procesamiento finalizado")
        self.process_button.setEnabled(False)
        self.progress.setValue(100)

        self.export_keypoints_button.setEnabled(bool(self.frame_results))
        self.export_anomalies_button.setEnabled(bool(self.anomalies))

        QMessageBox.information(
            self,
            "Procesamiento completado",
            "El vídeo se ha procesado correctamente.",
        )

    def reset_processing(self, clear_video=True):
        self.timer.stop()
        self.processing = False

        self.frame_results.clear()
        self.anomalies.clear()
        self.metrics.reset()
        self.anomaly_detector.reset()

        self.anomalies_table.setRowCount(0)
        self.progress.setValue(0)

        self.update_metrics()

        if self.detector is not None:
            self.detector.close()
            self.detector = None

        if self.current_video_path:
            self.video_manager.reset()
            self.process_button.setText("Procesar vídeo")
            self.process_button.setEnabled(True)

        self.export_keypoints_button.setEnabled(False)
        self.export_anomalies_button.setEnabled(False)

        if clear_video:
            self.video_label.setText("Carga un vídeo para comenzar")

    def export_keypoints(self):
        if not self.frame_results:
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar keypoints",
            "data/results/keypoints.csv",
            "CSV (*.csv)",
        )

        if not output_path:
            return

        export_keypoints_csv(self.frame_results, output_path)

        QMessageBox.information(
            self,
            "Exportación completada",
            f"Keypoints exportados en:\n{output_path}",
        )

    def export_anomalies(self):
        if not self.anomalies:
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar anomalías",
            "data/results/anomalias.csv",
            "CSV (*.csv)",
        )

        if not output_path:
            return

        export_anomalies_csv(self.anomalies, output_path)

        QMessageBox.information(
            self,
            "Exportación completada",
            f"Anomalías exportadas en:\n{output_path}",
        )

    def closeEvent(self, event):
        self.timer.stop()
        self.video_manager.release()

        if self.detector is not None:
            self.detector.close()

        event.accept()
