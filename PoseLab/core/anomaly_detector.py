import math
from collections import defaultdict


class AnomalyDetector:
    def __init__(
        self,
        confidence_threshold: float = 0.40,
        jump_threshold: float = 0.18,
    ):
        self.confidence_threshold = confidence_threshold
        self.jump_threshold = jump_threshold
        self.previous_keypoints = {}
        self.anomalies = []

    def reset(self):
        self.previous_keypoints.clear()
        self.anomalies.clear()

    def analyze_frame(self, result):
        frame_anomalies = []

        current = {kp.name: kp for kp in result.keypoints}

        for kp in result.keypoints:
            if kp.confidence < self.confidence_threshold:
                frame_anomalies.append(
                    {
                        "frame_id": result.frame_id,
                        "timestamp": result.timestamp,
                        "type": "baja_confianza",
                        "keypoint": kp.name,
                        "value": round(kp.confidence, 4),
                        "description": "Keypoint con confianza inferior al umbral definido.",
                    }
                )

            previous = self.previous_keypoints.get(kp.name)

            if previous is not None:
                distance = math.sqrt((kp.x - previous.x) ** 2 + (kp.y - previous.y) ** 2)

                if distance > self.jump_threshold:
                    frame_anomalies.append(
                        {
                            "frame_id": result.frame_id,
                            "timestamp": result.timestamp,
                            "type": "salto_brusco",
                            "keypoint": kp.name,
                            "value": round(distance, 4),
                            "description": "Desplazamiento excesivo del keypoint entre frames consecutivos.",
                        }
                    )

        if result.pose_detected:
            expected_keypoints = set(self.previous_keypoints.keys())

            for missing_name in expected_keypoints:
                if missing_name not in current:
                    frame_anomalies.append(
                        {
                            "frame_id": result.frame_id,
                            "timestamp": result.timestamp,
                            "type": "punto_perdido",
                            "keypoint": missing_name,
                            "value": 0,
                            "description": "Keypoint presente anteriormente y no detectado en el frame actual.",
                        }
                    )

        self.previous_keypoints = current
        self.anomalies.extend(frame_anomalies)

        return frame_anomalies
