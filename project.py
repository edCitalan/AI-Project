"""
YOLOv3 object detection demo (Windows-friendly script).

This is a cleaned-up version of a Colab notebook export.
It can:
- Download model files (yolov3.weights/cfg + coco.names)
- Run detection on one or more images
- Run detection on a webcam feed (press Q or ESC to quit)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.0f} TB"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def reporthook(blocknum: int, blocksize: int, totalsize: int) -> None:
        if totalsize <= 0:
            return
        downloaded = min(blocknum * blocksize, totalsize)
        pct = downloaded * 100 / totalsize
        sys.stdout.write(
            f"\r  {dest.name}: {pct:6.2f}% ({_format_bytes(downloaded)} / {_format_bytes(totalsize)})"
        )
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=reporthook)
        sys.stdout.write("\n")
        sys.stdout.flush()
        tmp.replace(dest)
    finally:
        if tmp.exists() and not dest.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def ensure_model_files(models_dir: Path) -> Dict[str, Path]:
    """Ensure YOLOv3 files exist locally, downloading if needed."""
    files = {
        "weights": ("https://pjreddie.com/media/files/yolov3.weights", models_dir / "yolov3.weights"),
        "cfg": (
            "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
            models_dir / "yolov3.cfg",
        ),
        "names": (
            "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
            models_dir / "coco.names",
        ),
    }

    print("Checking model files...")
    for _, (url, path) in files.items():
        if path.exists() and path.stat().st_size > 0:
            print(f"  [OK] {path.name} already exists")
            continue
        print(f"  Downloading {path.name} ...")
        download_file(url, path)
        print(f"  [OK] {path.name} downloaded")

    return {k: v[1] for k, v in files.items()}


def ensure_sample_image(dest: Path) -> Path:
    """Download a small sample image for a quick end-to-end test."""
    url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [OK] Sample image already exists: {dest}")
        return dest
    print("  Downloading sample image (dog.jpg) ...")
    download_file(url, dest)
    print(f"  [OK] Sample image downloaded: {dest}")
    return dest


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    time: str


class ObjectDetectionSystem:
    def __init__(
        self,
        weights_path: Path,
        cfg_path: Path,
        names_path: Path,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ):
        import cv2
        import numpy as np

        self.cv2 = cv2
        self.np = np

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        print("Loading YOLO model (OpenCV DNN)...")
        self.net = cv2.dnn.readNet(str(weights_path), str(cfg_path))

        with open(names_path, "r", encoding="utf-8") as f:
            self.classes = [line.strip() for line in f if line.strip()]

        layer_names = self.net.getLayerNames()
        unconnected = self.net.getUnconnectedOutLayers()
        # OpenCV returns either [200, 227, 254] or [[200], [227], [254]] depending on version.
        unconnected = [int(i) for i in self.np.array(unconnected).flatten().tolist()]
        self.output_layers = [layer_names[i - 1] for i in unconnected]

        rng = self.np.random.default_rng(42)
        self.colors = rng.uniform(0, 255, size=(len(self.classes), 3))

        self.detection_log: List[Detection] = []
        self.object_counts: Dict[str, int] = {}

        print("Model loaded successfully.")

    def detect_objects(self, frame):
        """Detect objects in a BGR frame."""
        cv2 = self.cv2
        np = self.np

        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids: List[int] = []
        confidences: List[float] = []
        boxes: List[List[int]] = []

        for out in outs:
            for detection in out:
                # YOLOv3 format: [cx, cy, w, h, objectness, class_scores...]
                objectness = float(detection[4])
                class_scores = detection[5:]
                class_id = int(np.argmax(class_scores))
                class_score = float(class_scores[class_id])
                confidence = objectness * class_score

                if confidence < self.conf_threshold:
                    continue

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        indexes = np.array(indexes).flatten().tolist() if len(indexes) else []
        return boxes, confidences, class_ids, indexes

    def draw_detections(self, frame, boxes, confidences, class_ids, indexes):
        cv2 = self.cv2
        detected: List[Detection] = []

        for i in indexes:
            x, y, w, h = boxes[i]
            label = self.classes[class_ids[i]] if 0 <= class_ids[i] < len(self.classes) else str(class_ids[i])
            confidence = float(confidences[i])
            color = self.colors[class_ids[i]] if 0 <= class_ids[i] < len(self.colors) else (0, 255, 0)
            color = tuple(int(c) for c in color)

            # Clamp to image bounds
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1] - 1, x + w), min(frame.shape[0] - 1, y + h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, label_text, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            det = Detection(label=label, confidence=confidence, time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            detected.append(det)
            self.object_counts[label] = self.object_counts.get(label, 0) + 1

        return frame, detected

    def save_log(self, filename: Path) -> None:
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write("=== OBJECT DETECTION LOG ===\n")
            f.write("Model: YOLOv3 (COCO Dataset)\n\n")
            for entry in self.detection_log:
                f.write(f"[{entry.time}] {entry.label} (confidence: {entry.confidence:.2f})\n")
            f.write("\n=== OBJECT STATISTICS (Cumulative) ===\n")
            for obj, count in sorted(self.object_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{obj}: {count} detections\n")
        print(f"Log saved to {filename}")

    def run_on_image(self, image_path: Path, outputs_dir: Path, show: bool) -> Tuple[Path, List[Detection]]:
        cv2 = self.cv2
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        boxes, confidences, class_ids, indexes = self.detect_objects(frame)
        frame, detected_objects = self.draw_detections(frame, boxes, confidences, class_ids, indexes)
        self.detection_log.extend(detected_objects)

        outputs_dir.mkdir(parents=True, exist_ok=True)
        output_path = outputs_dir / f"detected_{image_path.name}"
        cv2.imwrite(str(output_path), frame)

        print(f"\n[OK] {image_path.name}: {len(detected_objects)} objects")
        for obj in detected_objects:
            print(f"  - {obj.label}: {obj.confidence:.2f}")
        print(f"Saved: {output_path}")

        if show:
            cv2.imshow("Detection Result (press any key to close)", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return output_path, detected_objects

    def run_on_webcam(self, camera_index: int, outputs_dir: Path, show: bool) -> None:
        cv2 = self.cv2
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {camera_index}.")

        outputs_dir.mkdir(parents=True, exist_ok=True)
        log_every_n = 10
        prev_time, frame_count = time.time(), 0

        print("Starting live detection. Press 'Q' or 'ESC' to exit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes, confidences, class_ids, indexes = self.detect_objects(frame)
            frame, detected_objects = self.draw_detections(frame, boxes, confidences, class_ids, indexes)

            frame_count += 1
            elapsed = time.time() - prev_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            if detected_objects and (frame_count % log_every_n == 0):
                self.detection_log.extend(detected_objects)

            if show:
                cv2.imshow("YOLOv3 Webcam Detection (Q/ESC to quit)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            else:
                # Headless mode: stop after a short warmup.
                if frame_count >= 150:
                    break

        cap.release()
        if show:
            cv2.destroyAllWindows()
        print("Webcam detection stopped.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv3 object detection (image/webcam).")
    p.add_argument("--models-dir", default=".models", help="Directory for YOLO model files.")
    p.add_argument("--outputs-dir", default="outputs", help="Directory to write results/logs.")
    p.add_argument("--download-only", action="store_true", help="Only download model files, then exit.")

    mode = p.add_mutually_exclusive_group(required=False)
    mode.add_argument("--image", action="append", help="Image path(s). Can be provided multiple times.")
    mode.add_argument("--webcam", action="store_true", help="Run on webcam.")
    mode.add_argument("--sample", action="store_true", help="Download a sample image and run detection on it.")

    p.add_argument("--camera-index", type=int, default=0, help="Webcam index (default 0).")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default 0.5).")
    p.add_argument("--nms", type=float, default=0.4, help="NMS threshold (default 0.4).")
    p.add_argument("--show", action="store_true", help="Show OpenCV windows (image result / webcam feed).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    models_dir = Path(args.models_dir)
    outputs_dir = Path(args.outputs_dir)

    try:
        model_paths = ensure_model_files(models_dir)
    except Exception as e:
        print(f"[ERROR] Failed to download model files: {e}")
        return 2

    if args.download_only:
        print("[OK] Download complete.")
        return 0

    try:
        detector = ObjectDetectionSystem(
            weights_path=model_paths["weights"],
            cfg_path=model_paths["cfg"],
            names_path=model_paths["names"],
            conf_threshold=args.conf,
            nms_threshold=args.nms,
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize detector: {e}")
        print("Tip: install dependencies with: python -m pip install -r requirements.txt")
        return 2

    try:
        if args.webcam:
            detector.run_on_webcam(camera_index=args.camera_index, outputs_dir=outputs_dir, show=args.show)
        elif args.sample:
            sample_path = ensure_sample_image(outputs_dir / "sample_dog.jpg")
            detector.run_on_image(sample_path, outputs_dir=outputs_dir, show=args.show)
        else:
            images = args.image or []
            if not images:
                print("No mode selected. Use --image path.jpg (or multiple), --webcam, or --sample.")
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

