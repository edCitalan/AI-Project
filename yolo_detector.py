"""
Shared YOLOv3 detector module.

- Downloads YOLOv3 model files (weights/cfg + coco.names) into a models directory
- Provides ObjectDetectionSystem usable by image scripts, webcam demos, etc.

Notes (Windows):
- Avoids printing non-ASCII characters to prevent UnicodeEncodeError on some consoles.
"""

from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def _format_bytes(n: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        n: Number of bytes.

    Returns:
        Human-readable size string (e.g. "625 B", "8 KB", "237 MB").
    """
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.0f} TB"


def download_file(url: str, dest: Path) -> None:
    """Download a URL to a local file with a progress indicator.

    This writes to a temporary `.part` file first and then renames it to `dest`
    on success.

    Args:
        url: Source URL.
        dest: Destination file path.
    """
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
    """Ensure YOLOv3 files exist locally, downloading if needed.

    Args:
        models_dir: Directory where `yolov3.weights`, `yolov3.cfg`, and
            `coco.names` will be stored.

    Returns:
        A dict with keys `weights`, `cfg`, `names` mapping to the local file
        paths.
    """
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
    """Download a small sample image for a quick end-to-end test.

    The image is YOLO's standard `dog.jpg`.

    Args:
        dest: Where to write the sample image.

    Returns:
        The path to the sample image (same as `dest`).
    """
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
    """Single detection record for logging/UI.

    Attributes:
        label: COCO class label.
        confidence: Confidence score (0..1).
        time: Timestamp string when detection was recorded.
    """
    label: str
    confidence: float
    time: str


class ObjectDetectionSystem:
    """YOLOv3 detector using OpenCV DNN.

    This class loads YOLOv3 weights/config, performs inference on frames, and
    provides helpers to draw detections and write logs.
    """
    def __init__(
        self,
        weights_path: Path,
        cfg_path: Path,
        names_path: Path,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ):
        """Create a YOLOv3 detector instance.

        Args:
            weights_path: Path to `yolov3.weights`.
            cfg_path: Path to `yolov3.cfg`.
            names_path: Path to `coco.names`.
            conf_threshold: Minimum confidence score to keep a detection.
            nms_threshold: Non-max suppression threshold.
        """
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
        unconnected = [int(i) for i in self.np.array(unconnected).flatten().tolist()]
        self.output_layers = [layer_names[i - 1] for i in unconnected]

        rng = self.np.random.default_rng(42)
        self.colors = rng.uniform(0, 255, size=(len(self.classes), 3))

        self.detection_log: List[Detection] = []
        self.object_counts: Dict[str, int] = {}

        print("Model loaded successfully.")

    def detect_objects(self, frame):
        """Detect objects in a BGR frame.

        Args:
            frame: OpenCV BGR image (H x W x 3).

        Returns:
            Tuple of (boxes, confidences, class_ids, indexes) where:
            - boxes: list of [x, y, w, h] (pixel coordinates)
            - confidences: list of confidence floats
            - class_ids: list of integer class indices into `self.classes`
            - indexes: list of kept indices after NMS
        """
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
                # YOLOv3: [cx, cy, w, h, objectness, class_scores...]
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
        """Draw bounding boxes, labels, and update counts.

        Args:
            frame: OpenCV BGR image (modified in-place).
            boxes: Output from `detect_objects`.
            confidences: Output from `detect_objects`.
            class_ids: Output from `detect_objects`.
            indexes: Output from `detect_objects` (kept indices after NMS).

        Returns:
            (frame, detected) where `detected` is a list of `Detection`.
        """
        cv2 = self.cv2
        detected: List[Detection] = []

        for i in indexes:
            x, y, w, h = boxes[i]
            label = self.classes[class_ids[i]] if 0 <= class_ids[i] < len(self.classes) else str(class_ids[i])
            confidence = float(confidences[i])
            color = self.colors[class_ids[i]] if 0 <= class_ids[i] < len(self.colors) else (0, 255, 0)
            color = tuple(int(c) for c in color)

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1] - 1, x + w), min(frame.shape[0] - 1, y + h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label}: {confidence:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            det = Detection(label=label, confidence=confidence, time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            detected.append(det)
            self.object_counts[label] = self.object_counts.get(label, 0) + 1

        return frame, detected

    def save_log(self, filename: Path) -> None:
        """Write the accumulated detection log and counts to a text file.

        Args:
            filename: Path to write the log file (parent directories created).
        """
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
        """Run detection on a single image, write an annotated output image.

        Args:
            image_path: Input image path.
            outputs_dir: Output directory for annotated image.
            show: If True, show the resulting image in an OpenCV window.

        Returns:
            (output_path, detections)

        Raises:
            FileNotFoundError: If the image cannot be loaded by OpenCV.
        """
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


