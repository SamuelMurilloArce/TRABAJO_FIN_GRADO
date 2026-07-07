from dataclasses import dataclass
from typing import Optional


@dataclass
class Keypoint:
    frame_id: int
    timestamp: float
    model: str
    person_id: int
    name: str
    x: float
    y: float
    z: Optional[float]
    confidence: float


@dataclass
class PoseFrameResult:
    frame_id: int
    timestamp: float
    model: str
    keypoints: list[Keypoint]
    pose_detected: bool
    processing_time_ms: float
