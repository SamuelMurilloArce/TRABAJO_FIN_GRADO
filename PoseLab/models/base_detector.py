from abc import ABC, abstractmethod


class BasePoseDetector(ABC):
    name = "base"

    @abstractmethod
    def detect(self, frame_bgr, frame_id: int, timestamp: float):
        pass

    @abstractmethod
    def draw(self, frame_bgr, detection_result):
        pass

    def close(self):
        pass
