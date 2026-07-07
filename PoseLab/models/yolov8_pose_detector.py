import time
import cv2
from ultralytics import YOLO

from models.base_detector import BasePoseDetector
from core.keypoint_schema import Keypoint, PoseFrameResult


YOLO_POSE_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


class YOLOv8PoseDetector(BasePoseDetector):
    name = "YOLOv8n-pose"

    def __init__(self, model_path: str = "yolov8n-pose.pt", confidence: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame_bgr, frame_id: int, timestamp: float):
        start = time.perf_counter()

        results = self.model.predict(
            frame_bgr,
            conf=self.confidence,
            verbose=False,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        keypoints = []
        pose_detected = False

        if results and len(results) > 0:
            result = results[0]

            if result.keypoints is not None and result.keypoints.xy is not None:
                xy = result.keypoints.xy.cpu().numpy()
                conf = result.keypoints.conf.cpu().numpy()

                if len(xy) > 0:
                    pose_detected = True

                    person_id = 0
                    person_keypoints = xy[0]
                    person_confidences = conf[0]

                    height, width = frame_bgr.shape[:2]

                    for idx, point in enumerate(person_keypoints):
                        x_px, y_px = point

                        keypoints.append(
                            Keypoint(
                                frame_id=frame_id,
                                timestamp=timestamp,
                                model=self.name,
                                person_id=person_id,
                                name=YOLO_POSE_KEYPOINT_NAMES[idx],
                                x=float(x_px / width),
                                y=float(y_px / height),
                                z=0.0,
                                confidence=float(person_confidences[idx]),
                            )
                        )

        return PoseFrameResult(
            frame_id=frame_id,
            timestamp=timestamp,
            model=self.name,
            keypoints=keypoints,
            pose_detected=pose_detected,
            processing_time_ms=elapsed_ms,
        ), results

    def draw(self, frame_bgr, raw_result):
        if not raw_result or len(raw_result) == 0:
            return frame_bgr

        plotted = raw_result[0].plot()
        return plotted

    def close(self):
        self.model = None