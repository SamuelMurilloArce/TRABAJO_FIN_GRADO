from pathlib import Path
import pandas as pd


def export_keypoints_csv(frame_results, output_path: str):
    rows = []

    for result in frame_results:
        for kp in result.keypoints:
            rows.append(
                {
                    "frame_id": kp.frame_id,
                    "timestamp": kp.timestamp,
                    "model": kp.model,
                    "person_id": kp.person_id,
                    "keypoint": kp.name,
                    "x": kp.x,
                    "y": kp.y,
                    "z": kp.z,
                    "confidence": kp.confidence,
                }
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return output_path


def export_anomalies_csv(anomalies, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(anomalies)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return output_path
