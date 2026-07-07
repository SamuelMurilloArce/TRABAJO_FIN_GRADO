import time
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from models.base_detector import BasePoseDetector
from core.keypoint_schema import Keypoint, PoseFrameResult


MEDIAPIPE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


class MediaPipePoseDetector(BasePoseDetector):
    name = "MediaPipe Pose"

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 0,
    ):
        model_path = "models/pose_landmarker_lite.task"

        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame_bgr, frame_id: int, timestamp: float):
        start = time.perf_counter()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )

        timestamp_ms = int(timestamp * 1000)

        raw_result = self.detector.detect_for_video(mp_image, timestamp_ms)

        elapsed_ms = (time.perf_counter() - start) * 1000

        keypoints = []

        pose_detected = bool(raw_result.pose_landmarks)

        if pose_detected:
            landmarks = raw_result.pose_landmarks[0]

            for idx, landmark in enumerate(landmarks):
                name = MEDIAPIPE_LANDMARK_NAMES[idx]

                confidence = getattr(landmark, "visibility", None)
                if confidence is None:
                    confidence = getattr(landmark, "presence", 0.0)

                keypoints.append(
                    Keypoint(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        model=self.name,
                        person_id=0,
                        name=name,
                        x=float(landmark.x),
                        y=float(landmark.y),
                        z=float(landmark.z),
                        confidence=float(confidence),
                    )
                )

        return PoseFrameResult(
            frame_id=frame_id,
            timestamp=timestamp,
            model=self.name,
            keypoints=keypoints,
            pose_detected=pose_detected,
            processing_time_ms=elapsed_ms,
        ), raw_result

    def draw(self, frame_bgr, raw_result):
        output = frame_bgr.copy()

        if not raw_result.pose_landmarks:
            return output

        height, width, _ = output.shape
        landmarks = raw_result.pose_landmarks[0]

        connections = [
            (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16),
            (11, 23), (12, 24),
            (23, 24),
            (23, 25), (25, 27),
            (24, 26), (26, 28),
            (27, 29), (29, 31),
            (28, 30), (30, 32),
        ]

        for start_idx, end_idx in connections:
            p1 = landmarks[start_idx]
            p2 = landmarks[end_idx]

            x1, y1 = int(p1.x * width), int(p1.y * height)
            x2, y2 = int(p2.x * width), int(p2.y * height)

            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for landmark in landmarks:
            x, y = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(output, (x, y), 4, (0, 0, 255), -1)

        return output

    def close(self):
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()
            self.detector = None