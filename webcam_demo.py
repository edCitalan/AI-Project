"""
Live webcam demo (kept separate from project.py).

Usage (PowerShell):
  cd "C:\\Users\\edwar\\OneDrive\\Desktop\\New folder"
  python webcam_demo.py --camera 0

Press Q or ESC to quit.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from yolo_detector import ObjectDetectionSystem, ensure_model_files


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the webcam demo.

    Returns:
        Parsed argparse namespace.
    """
    p = argparse.ArgumentParser(description="YOLOv3 live webcam demo.")
    p.add_argument("--camera", type=int, default=0, help="Webcam index (default 0).")
    p.add_argument("--models-dir", default=".models", help="Directory for YOLO model files.")
    p.add_argument("--outputs-dir", default="outputs", help="Directory to write logs.")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default 0.5).")
    p.add_argument("--nms", type=float, default=0.4, help="NMS threshold (default 0.4).")
    return p.parse_args()


def main() -> int:
    """Run the live webcam demo loop and write a log file on exit.

    Returns:
        Process exit code (0 success, 2 on webcam open failure).
    """
    args = parse_args()
    models_dir = Path(args.models_dir)
    outputs_dir = Path(args.outputs_dir)

    model_paths = ensure_model_files(models_dir)
    detector = ObjectDetectionSystem(
        weights_path=model_paths["weights"],
        cfg_path=model_paths["cfg"],
        names_path=model_paths["names"],
        conf_threshold=args.conf,
        nms_threshold=args.nms,
    )

    cv2 = detector.cv2
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam index {args.camera}")
        return 2

    print("Starting live object detection. Press 'Q' or 'ESC' to exit.")
    prev_time = time.time()
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        boxes, confidences, class_ids, indexes = detector.detect_objects(frame)
        frame, detected_objects = detector.draw_detections(frame, boxes, confidences, class_ids, indexes)

        current_frame_counts: Dict[str, int] = {}
        for det in detected_objects:
            current_frame_counts[det.label] = current_frame_counts.get(det.label, 0) + 1

        frame_count += 1
        elapsed_time = time.time() - prev_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(
            frame,
            "Platform: Jetson (Emulated)",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        y = 60
        for label, count in sorted(current_frame_counts.items(), key=lambda x: x[0]):
            cv2.putText(frame, f"Count ({label}): {count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            y += 28

        detector.detection_log.extend(detected_objects)

        cv2.imshow("LIVE YOLOv3 DETECTION (Q/ESC to stop)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    outputs_dir.mkdir(parents=True, exist_ok=True)
    log_path = outputs_dir / f"webcam_detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    detector.save_log(log_path)
    print(f"[OK] Total detections logged: {len(detector.detection_log)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


