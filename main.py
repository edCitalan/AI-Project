"""
Main entrypoint (kept separate from project.py).

One runner that calls ObjectDetectionSystem for:
- --sample (quick proof it works)
- --image (your own images)
- --webcam (live window, Q/ESC to quit)
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from yolo_detector import ObjectDetectionSystem, ensure_model_files, ensure_sample_image


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `main.py`.

    Returns:
        Parsed argparse namespace.
    """
    p = argparse.ArgumentParser(description="YOLOv3 object detection (main entrypoint).")
    p.add_argument("--models-dir", default=".models", help="Directory for YOLO model files.")
    p.add_argument("--outputs-dir", default="outputs", help="Directory to write results/logs.")
    p.add_argument("--download-only", action="store_true", help="Only download model files, then exit.")

    mode = p.add_mutually_exclusive_group(required=False)
    mode.add_argument("--sample", action="store_true", help="Download a sample image and run detection on it.")
    mode.add_argument("--image", action="append", help="Image path(s). Can be provided multiple times.")
    mode.add_argument("--webcam", action="store_true", help="Run on webcam (opens a window).")

    p.add_argument("--camera-index", type=int, default=0, help="Webcam index (default 0).")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default 0.5).")
    p.add_argument("--nms", type=float, default=0.4, help="NMS threshold (default 0.4).")
    p.add_argument("--show", action="store_true", help="Show OpenCV windows for image results or webcam.")
    return p.parse_args()


def run_webcam(detector: ObjectDetectionSystem, camera_index: int, show: bool) -> None:
    """Run live detection on a webcam feed.

    Args:
        detector: Initialized YOLO detector.
        camera_index: OpenCV camera index (usually 0 or 1).
        show: If True, display a window with the live feed.

    Raises:
        RuntimeError: If the webcam cannot be opened.
    """
    cv2 = detector.cv2
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera_index}.")

    print("Starting live detection. Press 'Q' or 'ESC' to exit.")
    prev_time, frame_count = time.time(), 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, confidences, class_ids, indexes = detector.detect_objects(frame)
        frame, detected_objects = detector.draw_detections(frame, boxes, confidences, class_ids, indexes)
        detector.detection_log.extend(detected_objects)

        frame_count += 1
        elapsed = time.time() - prev_time
        fps = frame_count / elapsed if elapsed > 0 else 0.0
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if show:
            cv2.imshow("YOLOv3 Webcam Detection (Q/ESC to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
        else:
            # Headless safety: don't loop forever.
            if frame_count >= 150:
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()
    print("Webcam detection stopped.")


def main() -> int:
    """Program entrypoint.

    Returns:
        Process exit code:
        - 0 for success
        - 1 for CLI usage errors (no mode selected)
        - 2 for runtime failures
    """
    args = parse_args()
    models_dir = Path(args.models_dir)
    outputs_dir = Path(args.outputs_dir)

    model_paths = ensure_model_files(models_dir)
    if args.download_only:
        print("[OK] Download complete.")
        return 0

    detector = ObjectDetectionSystem(
        weights_path=model_paths["weights"],
        cfg_path=model_paths["cfg"],
        names_path=model_paths["names"],
        conf_threshold=args.conf,
        nms_threshold=args.nms,
    )

    try:
        if args.webcam:
            run_webcam(detector, camera_index=args.camera_index, show=args.show)
        elif args.sample:
            sample_path = ensure_sample_image(outputs_dir / "sample_dog.jpg")
            detector.run_on_image(sample_path, outputs_dir=outputs_dir, show=args.show)
        else:
            images = args.image or []
            if not images:
                print("No mode selected. Use --sample, --image path.jpg (or multiple), or --webcam.")
                return 1
            for img in images:
                detector.run_on_image(Path(img), outputs_dir=outputs_dir, show=args.show)

        log_name = f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        detector.save_log(outputs_dir / log_name)
        return 0
    except Exception as e:
        print(f"[ERROR] Runtime error: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


