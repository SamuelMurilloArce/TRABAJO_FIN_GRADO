# PoseLab

PoseLab es una aplicación de escritorio desarrollada en Python para el análisis del movimiento humano mediante técnicas de estimación de pose.

El objetivo del proyecto es proporcionar una herramienta que permita procesar vídeos, detectar la pose humana, visualizar el esqueleto generado por diferentes modelos de Inteligencia Artificial y calcular métricas biomecánicas que puedan servir de apoyo al estudio de enfermedades neurodegenerativas como la Esclerosis Lateral Amiotrófica (ELA).

Este proyecto ha sido desarrollado como Trabajo Fin de Grado en Ingeniería Informática.

---

## Características

- Carga de vídeos desde el equipo.
- Procesamiento fotograma a fotograma.
- Selección de diferentes modelos de estimación de pose.
- Visualización del esqueleto sobre el vídeo.
- Extracción de keypoints articulares.
- Cálculo de métricas biomecánicas.
- Detección de anomalías durante el procesamiento.
- Exportación de resultados en formato CSV y JSON.
- Arquitectura modular preparada para incorporar nuevos modelos.

---

## Tecnologías utilizadas

- Python 3.10
- PySide6
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Ultralytics (YOLOv8 Pose)
- PyTorch

---

## Estructura del proyecto

```
PoseLab/
│
├── app/
│   ├── styles/
│   ├── widgets/
│   └── main_window.py
│
├── core/
│   ├── anomaly_detector.py
│   ├── exporters.py
│   ├── keypoint_schema.py
│   ├── metrics.py
│   └── video_manager.py
│
├── models/
│   ├── base_detector.py
│   └── yolov8_pose_detector.py
│
├── assets/
├── data/
│   ├── input_videos/
│   └── results/
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Instalación

Clonar el repositorio

```bash
git clone https://github.com/SamuelMurilloArce/TRABAJO_FIN_GRADO
```

Acceder al proyecto

```bash
cd PoseLab
```

Crear un entorno virtual

```bash
python -m venv .venv
```

Activarlo

Windows

```bash
.venv\Scripts\activate
```

Linux

```bash
source .venv/bin/activate
```

Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Ejecución

```bash
python main.py
```

---

## Flujo de trabajo

1. Cargar un vídeo.
2. Seleccionar el modelo de estimación de pose.
3. Iniciar el procesamiento.
4. Visualizar el esqueleto detectado.
5. Analizar las métricas obtenidas.
6. Exportar los resultados.

---

## Resultados exportados

La aplicación permite exportar:

- Coordenadas de keypoints (CSV).
- Métricas calculadas.
- Registro de anomalías (JSON).

---

## Futuras mejoras

- Comparación automática entre varios modelos.
- Exportación de informes PDF.
- Incorporación de nuevos modelos de estimación de pose.
- Representación gráfica de métricas.
- Procesamiento en tiempo real mediante webcam.

---

## Autor

Samuel Murillo Arce

Trabajo Fin de Grado

Grado en Ingeniería Informática