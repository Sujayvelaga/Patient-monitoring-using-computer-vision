# IV Bottle YOLOv8 Fine-Tuning Instructions

## Goal
Fine-tune YOLOv8n to detect IV saline bottles specifically, improving accuracy over the generic COCO "bottle" class.

---

## Step 1: Collect Images (200-500 images)

### Method A: Camera Capture
```python
"""Capture training images from your ESP32-CAM or webcam."""
import cv2
import os
import time

os.makedirs("iv_bottle_dataset/images", exist_ok=True)

cap = cv2.VideoCapture(0)  # or ESP32 URL
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Capture - Press SPACE to save, Q to quit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        path = f"iv_bottle_dataset/images/iv_{count:04d}.jpg"
        cv2.imwrite(path, frame)
        count += 1
        print(f"Saved: {path} ({count} images)")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Total: {count} images captured")
```

### Guidelines
- Capture from **multiple angles** (front, side, angled)
- Vary **lighting** (bright, dim, shadows, backlit)
- Include **different fluid levels** (full, half, low, empty)
- Include **both** IV bottle present and absent scenes
- Include scenes with **multiple objects** (bed, patient, equipment)

---

## Step 2: Annotate with Roboflow (Free)

1. Go to [roboflow.com](https://roboflow.com) — create free account
2. Create new project → Object Detection → Name: "IV Bottle Detection"
3. Upload your images
4. Annotate each image:
   - Draw bounding box around the **IV bottle** → label: `iv_bottle`
   - Draw bounding box around the **fluid region** → label: `iv_fluid` (optional)
5. Apply augmentations:
   - Brightness: -25% to +25%
   - Rotation: -15° to +15°
   - Blur: up to 2.5px
   - Noise: up to 3%
6. Export:
   - Format: **YOLOv8**
   - Download ZIP

---

## Step 3: Organize Dataset

After downloading from Roboflow, your structure should be:
```
iv_bottle_dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### data.yaml content:
```yaml
train: ./train/images
val: ./valid/images
test: ./test/images

nc: 1  # number of classes (or 2 if including iv_fluid)
names: ['iv_bottle']  # or ['iv_bottle', 'iv_fluid']
```

---

## Step 4: Fine-Tune YOLOv8

```python
from ultralytics import YOLO

# Load pretrained YOLOv8n
model = YOLO("yolov8n.pt")

# Fine-tune on IV bottle dataset
results = model.train(
    data="iv_bottle_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,       # Reduce to 8 if GPU memory is limited
    lr0=0.001,
    patience=15,    # Early stopping
    project="runs/iv_bottle",
    name="yolov8n_iv",
    pretrained=True,
    device="",      # Auto-select GPU/CPU
)

# Validate
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")

# Export best model
best_model_path = "runs/iv_bottle/yolov8n_iv/weights/best.pt"
print(f"Best model saved: {best_model_path}")
```

---

## Step 5: Integrate into System

Update `config.py`:
```python
@dataclass
class YOLOConfig:
    model_path: str = "runs/iv_bottle/yolov8n_iv/weights/best.pt"
    # ... rest stays the same
```

Or pass via command line:
```bash
python main.py --cam 0
# The system will use the model path from config.py
```

---

## Step 6: Test Detection

```python
from ultralytics import YOLO
import cv2

model = YOLO("runs/iv_bottle/yolov8n_iv/weights/best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)
    annotated = results[0].plot()

    cv2.imshow("IV Bottle Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

---

## Tips for Best Results
- **Minimum 200 images** for decent results, 500+ for robust detection
- **Include negatives**: images WITHOUT IV bottles (20-30% of dataset)
- **Augment heavily**: Roboflow augmentations increase effective dataset 3-5x
- **Monitor training**: Check `runs/iv_bottle/yolov8n_iv/results.png` for loss curves
- **Test on live camera**: Ensure it works with your actual ESP32-CAM feed
- **Expected metrics**: mAP50 > 0.85 is good, > 0.92 is excellent
