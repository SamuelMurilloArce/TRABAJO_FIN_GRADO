import time
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


class YOLOPoseDetector(BasePoseDetector):
    name = "YOLO Pose"

    def __init__(self, model_path="yolov8n-pose.pt", confidence=0.5):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame_bgr, frame_id: int, timestamp: float):
        start = time.perf_counter()

        results = self.model(frame_bgr, conf=self.confidence, verbose=False)

        elapsed_ms = (time.perf_counter() - start) * 1000

        keypoints = []
        pose_detected = False

        if results and len(results) > 0:
            result = results[0]

            if result.keypoints is not None:
                xy = result.keypoints.xy.cpu().numpy()

                conf = None
                if result.keypoints.conf is not None:
                    conf = result.keypoints.conf.cpu().numpy()

                height, width = frame_bgr.shape[:2]

                if len(xy) > 0:
                    pose_detected = True

                    for person_id, person_keypoints in enumerate(xy):
                        for idx, point in enumerate(person_keypoints):
                            if idx >= len(YOLO_POSE_KEYPOINT_NAMES):
                                continue

                            x_px, y_px = point

                            confidence = 0.0
                            if conf is not None:
                                confidence = float(conf[person_id][idx])

                            keypoints.append(
                                Keypoint(
                                    frame_id=frame_id,
                                    timestamp=timestamp,
                                    model=self.model_path,
                                    person_id=person_id,
                                    name=YOLO_POSE_KEYPOINT_NAMES[idx],
                                    x=float(x_px / width),
                                    y=float(y_px / height),
                                    z=0.0,
                                    confidence=confidence,
                                )
                            )

        return PoseFrameResult(
            frame_id=frame_id,
            timestamp=timestamp,
            model=self.model_path,
            keypoints=keypoints,
            pose_detected=pose_detected,
            processing_time_ms=elapsed_ms,
        ), results

    def draw(self, frame_bgr, raw_result):
        if raw_result and len(raw_result) > 0:
            return raw_result[0].plot()

        return frame_bgr

    def close(self):
        self.model = None