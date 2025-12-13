## Jetson (Emulated) YOLOv3 Object Detection

This repo runs YOLOv3 object detection using OpenCV DNN.

### Project Context

This project was built for an **Intelligent Systems (AI Term Project)** and is designed to be deployable on (or easily portable to) an **NVIDIA Jetson Nano Orin** environment (hardware or emulator).

### Setup

```powershell
cd "C:\Users\edwar\OneDrive\Desktop\New folder"
python -m pip install -r requirements.txt
```

### Download model files (one-time)

```powershell
python main.py --download-only
```

### Quick test (sample image)

```powershell
python main.py --sample
```

Outputs are written to `outputs\` (ignored by git).

### Run on your own image(s)

```powershell
python main.py --image "C:\full\path\to\photo.jpg"
```

### Live webcam demo

```powershell
python webcam_demo.py --camera 0
```

Press **Q** or **ESC** to quit.

---

## Term Project Technical Report: AI-Based Traffic Flow Analysis and Monitoring System

**Course**: Intelligent Systems (AI Term Project)  
**Due Date**: December 5th  
**Team Leader**: Edward Aguilar Citalan, Jefferson Pinargote, Lynn Khaing, Nicholas Frangione, and Santosh Parajuli
**Project Goal**: Design and build an intelligent system using core AI techniques (Vision, Learning, Reasoning) and deploy it to a simulated NVIDIA Jetson Nano Orin environment.

### Abstract

This report details the design and implementation of an AI-based traffic Flow Analysis and Monitoring System. The intelligent system utilizes a state-of-the-art Convolutional Neural Network (CNN), specifically the YOLOv3 model, to perform real-time object detection and classification of vehicles in simulated video streams. The project fulfills the requirement for an edge-AI system by simulating deployment on the NVIDIA Jetson Nano Orin platform, which is optimized for low-latency inference. Key AI techniques employed include Deep Learning (Learning) for model generation, object localization and classification (Vision), and Non-Max Suppression (NMS) (Reasoning) for accurate bounding box selection. The system successfully classifies common traffic objects (e.g., car, truck, bus) and generates time-stamped log files for traffic analysis.

### 1. Introduction and Objectives

#### 1.1 Problem Statement

Modern urban planning and traffic management require accurate, real-time data on traffic flow without the cost and privacy concerns associated with human monitoring. Deploying computationally intensive deep learning models directly on roadside equipment ("at the edge") is necessary to minimize network latency and server costs.

#### 1.2 Project Objective

The primary objective of this term project is to develop a functional, intelligent system that performs two key tasks:

- Real-Time Object Detection and Classification: Accurately identify and label vehicles (cars, trucks, buses) within a video stream using a high-performance deep learning model.
- Edge AI Deployment Simulation: Implement the system using a computational stack (Python, OpenCV DNN) that is directly translatable to the NVIDIA Jetson Nano Orin, demonstrating the feasibility of edge deployment.

### 2. Methodology and Core AI Techniques

The system architecture follows a classic computer vision pipeline: model loading, pre-processing, inference, and post-processing.

#### 2.1 Model Selection: YOLOv3

The YOLOv3 (You Only Look Once) architecture was selected as the core intelligence model.

- Learning: YOLOv3 is a Single-Shot Detector pre-trained on the COCO (Common Objects in Context) dataset. This model leverages transfer learning, using the knowledge base of 80 classes to achieve high accuracy without requiring further training time. The pre-computed weights (`yolov3.weights`) represent the culmination of this learning phase.
- Performance: YOLOv3 balances detection accuracy with inference speed, making it suitable for the resource-constrained environment of the Jetson Nano Orin.

#### 2.2 Inference and Vision Pipeline

The `detect_objects` function manages the inference pipeline:

- Input Preprocessing: The input image frame is converted into a 416x416-pixel blob using `cv2.dnn.blobFromImage`. This step normalizes pixel values and resizes the image to the fixed input dimension required by the YOLOv3 network.
- Forward Pass (Inference): The blob is passed through the CNN via `self.net.forward()`. This is the core computation step, where the learned features extract spatial and categorical information. The output is a matrix of raw predictions containing potential bounding box coordinates and confidence scores for all 80 COCO classes.

#### 2.3 Reasoning: Non-Max Suppression (NMS)

The Reasoning requirement is fulfilled by the Non-Max Suppression (NMS) algorithm, which cleans the raw model output.

- Function: The CNN often generates multiple overlapping bounding boxes for a single object. NMS systematically filters these duplicates.
- Mechanism: NMS evaluates predictions based on two metrics:
  - Confidence Score: Prioritizing the box with the highest confidence.
  - Intersection over Union (\(IoU\)): Calculating the overlap between the current box and all others. Boxes with an \(IoU\) exceeding a set threshold (e.g., 0.4) are suppressed (removed), leaving only the most accurate prediction.

This reasoning process ensures a clean, single bounding box is displayed for each detected object, providing a clear and reliable result.

### 3. Deployment and System Implementation

#### 3.1 Jetson Nano Orin Emulation

The system is designed for hardware deployment on the Jetson Nano Orin, an edge AI device featuring an integrated GPU and Deep Learning Accelerators (DLAs).

- Simulation Environment: The OpenCV DNN module is used for processing. This simulation is critical because:
  - It uses a common development stack that easily ports to the Jetson ecosystem.
  - It focuses on the same inference pipeline logic used by the native Jetson `detectNet` API, fulfilling the project's hardware mandate.

- Performance Discussion: While running YOLOv3 via the Python/OpenCV bridge is functional, real-world constraints dictate further optimization. For maximum Frames Per Second (FPS) on the Jetson, the model would typically be converted to a TensorRT engine to leverage half-precision floating-point (FP16) or integer (INT8) math, dramatically increasing throughput for true real-time video processing (often achieving \(10-20\) FPS compared to \(\le 5\) FPS without optimization).

#### 3.2 Post-Processing and Logging

The system provides robust post-processing features essential for a monitoring application:

- Visualization: Detected objects are labeled with their class name and confidence score, and visualized with a colored bounding box using `cv2.rectangle` and `cv2.putText`.
- Data Persistence: The system maintains an in-memory `detection_log` and `object_counts`. This data is saved to a file (`detection_log.txt`) upon completion, which provides quantifiable metrics for traffic analysis.

### 4. Results and Discussion

#### 4.1 Test Run Analysis

The system was tested using three different image files simulating real-world scenarios.

| Input Image | Detected Objects (Confidence ≥ 0.5) | Total Detections |
| --- | --- | --- |
| images (1).jpg | Car, Person, Traffic Light | 3 |
| 1942_BHG...jpg | Chair, Book, Vase | 3 |
| images.jpg | Truck, Car, Person | 3 |

**Total Logged Events**: **9**

The results demonstrate the model's accuracy in correctly classifying objects across diverse environments (indoor and outdoor). Critically, the use of Non-Max Suppression ensured that despite the model potentially outputting many overlapping boxes, the final frame annotations were clean and reliable.

#### 4.2 Traffic Flow Analysis Metrics

The generated `detection_log.txt` provides the following statistical summary:

| Object Class | Count | Purpose in Traffic System |
| --- | ---:| --- |
| Car | 2 | Primary vehicle count, flow rate. |
| Person | 2 | Pedestrian safety and crossing metrics. |
| Truck | 1 | Commercial vehicle density and routing. |
| Traffic Light | 1 | Contextual understanding of traffic signals. |
| (Other Classes) | 3 | Indoor object (Noise/Context) |

This quantifiable output proves the intelligent system's ability to act as a data-gathering sensor for traffic analysis, moving beyond simple detection into the domain of actionable intelligence.

### 5. Conclusion and Future Work

The AI-Based Traffic Flow Analysis and Monitoring System successfully meets the core requirements of the term project by integrating Vision (YOLOv3), Learning (pre-trained weights), and Reasoning (NMS) into a functional application designed for edge deployment. The current implementation provides accurate detection and comprehensive logging, which forms a solid foundation for a real-world system.

Future work should focus on:

- Optimization: Implement the TensorRT conversion pipeline to maximize FPS on the Jetson Orin hardware, enabling true real-time video stream processing.
- Tracking: Integrate a multi-object tracking algorithm (e.g., DeepSORT) to assign unique IDs to each detected object and accurately count traffic flow over time.
- Custom Data: Fine-tune the YOLO model (transfer learning) on a custom dataset of campus-specific vehicles (e.g., security carts, maintenance vehicles) to improve domain-specific accuracy.

### Appendix A: Source Code

The full source code is included in this repository (model download, `ObjectDetectionSystem`, and runnable demos).


