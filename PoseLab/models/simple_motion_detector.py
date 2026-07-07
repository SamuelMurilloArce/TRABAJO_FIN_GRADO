import time
import cv2
import numpy as np

from models.base_detector import BasePoseDetector
from core.keypoint_schema import Keypoint, PoseFrameResult


class SimpleMotionDetector(BasePoseDetector):
    name = "Detector simple de movimiento"

    def __init__(self, min_area: int = 800):
        self.previous_gray = None
        self.min_area = min_area

    def detect(self, frame_bgr, frame_id: int, timestamp: float):
        start = time.perf_counter()

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        keypoints = []
        pose_detected = False

        if self.previous_gray is not None:
            diff = cv2.absdiff(self.previous_gray, gray)
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)

                if area > self.min_area:
                    pose_detected = True
                    x, y, w, h = cv2.boundingRect(contour)

                    height, width = frame_bgr.shape[:2]

                    points = {
                        "head": (x + w / 2, y),
                        "left_shoulder": (x + w * 0.25, y + h * 0.25),
                        "right_shoulder": (x + w * 0.75, y + h * 0.25),
                        "center": (x + w / 2, y + h * 0.5),
                        "left_hip": (x + w * 0.35, y + h * 0.7),
                        "right_hip": (x + w * 0.65, y + h * 0.7),
                        "left_foot": (x + w * 0.35, y + h),
                        "right_foot": (x + w * 0.65, y + h),
                    }

                    for name, (px, py) in points.items():
                        keypoints.append(
                            Keypoint(
                                frame_id=frame_id,
                                timestamp=timestamp,
                                model=self.name,
                                person_id=0,
                                name=name,
                                x=float(px / width),
                                y=float(py / height),
                                z=0.0,
                                confidence=1.0,
                            )
                        )

        self.previous_gray = gray

        elapsed_ms = (time.perf_counter() - start) * 1000

        return PoseFrameResult(
            frame_id=frame_id,
            timestamp=timestamp,
            model=self.name,
            keypoints=keypoints,
            pose_detected=pose_detected,
            processing_time_ms=elapsed_ms,
        ), keypoints

    def draw(self, frame_bgr, raw_result):
        output = frame_bgr.copy()

        if not raw_result:
            return output

        height, width = output.shape[:2]

        points = {}
        for kp in raw_result:
            px = int(kp.x * width)
            py = int(kp.y * height)
            points[kp.name] = (px, py)

            cv2.circle(output, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(
                output,
                kp.name,
                (px + 5, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        connections = [
            ("head", "center"),
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "center"),
            ("right_shoulder", "center"),
            ("center", "left_hip"),
            ("center", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_foot"),
            ("right_hip", "right_foot"),
        ]

        for a, b in connections:
            if a in points and b in points:
                cv2.line(output, points[a], points[b], (0, 255, 0), 2)

        return output

    def close(self):
        self.previous_gray = None