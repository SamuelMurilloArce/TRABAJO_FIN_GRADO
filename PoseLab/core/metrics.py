import numpy as np


class MetricsCalculator:
    def __init__(self):
        self.frame_results = []

    def add_result(self, result):
        self.frame_results.append(result)

    def reset(self):
        self.frame_results.clear()

    def summary(self):
        if not self.frame_results:
            return {
                "total_frames": 0,
                "valid_frames": 0,
                "valid_frame_ratio": 0.0,
                "mean_confidence": 0.0,
                "mean_processing_time_ms": 0.0,
                "estimated_fps": 0.0,
                "mean_keypoints_per_frame": 0.0,
            }

        total_frames = len(self.frame_results)
        valid_frames = sum(1 for r in self.frame_results if r.pose_detected)

        confidences = [
            kp.confidence
            for r in self.frame_results
            for kp in r.keypoints
        ]

        processing_times = [r.processing_time_ms for r in self.frame_results]
        keypoints_per_frame = [len(r.keypoints) for r in self.frame_results]

        mean_processing = float(np.mean(processing_times)) if processing_times else 0.0

        return {
            "total_frames": total_frames,
            "valid_frames": valid_frames,
            "valid_frame_ratio": valid_frames / total_frames if total_frames else 0.0,
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "mean_processing_time_ms": mean_processing,
            "estimated_fps": 1000.0 / mean_processing if mean_processing > 0 else 0.0,
            "mean_keypoints_per_frame": float(np.mean(keypoints_per_frame)) if keypoints_per_frame else 0.0,
        }
