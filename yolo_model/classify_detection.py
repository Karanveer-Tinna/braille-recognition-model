import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO
from pathlib import Path
from model_cnn import BrailleCNN  # if defined in another file
import os

# --- Load YOLO model for object detection ---
yolo_model = YOLO('braille_model_2.pt')  # your trained YOLOv8 detection model

# --- Load Braille classification model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
braille_classifier = BrailleCNN(num_classes=64)  # adjust if your CNN uses 26 classes
braille_classifier.load_state_dict(torch.load("braille_cnn_model.pt", map_location=device))  # path to your trained CNN
braille_classifier.to(device)
braille_classifier.eval()

# --- Transform for CNN ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# --- Preprocessing for CNN (same as training) ---
def preprocess_for_cnn(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    denoised = cv.bilateralFilter(equalized, 9, 100, 100)
    thresh = cv.adaptiveThreshold(denoised, 255,
                                  cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY_INV,
                                  blockSize=5, C=2)
    return thresh

# --- Inference function ---
def detect_and_classify(image_path):
    image = cv.imread(str(image_path))
    if image is None:
        print(f"Error reading image: {image_path}")
        return

    detections = yolo_model.predict(source=str(image_path), save=False, verbose=False)[0]

    for i, box in enumerate(detections.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        processed = preprocess_for_cnn(cropped)
        try:
            processed = transform(processed)
        except:
            continue

        if processed.shape != (1, 28, 28):
            continue

        input_tensor = processed.unsqueeze(0).to(device)
        with torch.no_grad():
            output = braille_classifier(input_tensor)
            predicted_class = output.argmax(dim=1).item()

        print(f"Box {i+1}: Predicted Braille class = {predicted_class} | Coordinates = ({x1}, {y1}), ({x2}, {y2})")

# --- Run inference on a test folder ---
test_dir = Path(r"C:\Users\veerk\Downloads\braille.v2i.yolov11\test\images")
test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))

for img_path in test_images:
    print(f"\nProcessing {img_path.name}")
    detect_and_classify(img_path)
