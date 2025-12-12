## Jetson (Emulated) YOLOv3 Object Detection

This repo runs YOLOv3 object detection using OpenCV DNN.

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


