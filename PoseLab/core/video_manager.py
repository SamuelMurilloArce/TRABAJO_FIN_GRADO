import cv2


class VideoManager:
    def __init__(self):
        self.capture = None
        self.video_path = None
        self.fps = 0
        self.total_frames = 0
        self.width = 0
        self.height = 0

    def open(self, video_path: str):
        self.release()

        self.video_path = video_path
        self.capture = cv2.VideoCapture(video_path)

        if not self.capture.isOpened():
            raise RuntimeError("No se ha podido abrir el vídeo seleccionado.")

        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        if self.capture is None:
            return False, None

        return self.capture.read()

    def reset(self):
        if self.capture is not None:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def current_frame_index(self):
        if self.capture is None:
            return 0
        return int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))

    def timestamp_for_frame(self, frame_id: int):
        if self.fps <= 0:
            return 0.0
        return frame_id / self.fps

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
