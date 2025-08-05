import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO
from pathlib import Path
import os
from model_cnn import BrailleCNN

# --- Load YOLO model for object detection ---
yolo_model = YOLO('braille_model_2.pt')  # your trained YOLOv8 detection model

# --- Load Braille classification model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
braille_classifier = BrailleCNN(num_classes=26)  # adjust if your CNN uses 26 classes
braille_classifier.load_state_dict(torch.load("yolo_classify_model.pt", map_location=device))  # path to your trained CNN
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

# --- Function to parse YOLO label files ---
def parse_yolo_labels(label_path, img_width, img_height):
    ground_truths = []
    if not os.path.exists(label_path):
        return ground_truths
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            ground_truths.append({'class_id': class_id, 'box': [x1, y1, x2, y2]})
    return ground_truths

# --- Function to calculate Intersection over Union (IoU) ---
def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
        
    iou = inter_area / union_area
    return iou

# --- Evaluation function ---
def evaluate_and_classify(image_path, label_path):
    image = cv.imread(str(image_path))
    if image is None:
        print(f"Error reading image: {image_path}")
        return 0, 0

    img_height, img_width, _ = image.shape
    ground_truths = parse_yolo_labels(label_path, img_width, img_height)
    detections = yolo_model.predict(source=str(image_path), save=False, verbose=False)[0]

    correct_predictions = 0
    total_ground_truths = len(ground_truths)

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

        # Match detection with ground truth
        best_iou = 0
        matched_gt = None
        for gt in ground_truths:
            iou = calculate_iou([x1, y1, x2, y2], gt['box'])
            if iou > best_iou:
                best_iou = iou
                matched_gt = gt

        if matched_gt and best_iou > 0.5:  # IoU threshold of 0.5
            if predicted_class == matched_gt['class_id']:
                correct_predictions += 1
                print(f"  - Box {i+1}: Predicted class = {predicted_class}, Ground Truth = {matched_gt['class_id']} -> Correct")
            else:
                print(f"  - Box {i+1}: Predicted class = {predicted_class}, Ground Truth = {matched_gt['class_id']} -> Incorrect")
            # Remove matched ground truth to avoid double matching
            ground_truths.remove(matched_gt)

    return correct_predictions, total_ground_truths

# --- Run evaluation on the test folder ---
test_dir = Path(r"C:\Users\veerk\Downloads\braille.v2i.yolov11\test\images")
label_dir = Path(r"C:\Users\veerk\Downloads\braille.v2i.yolov11\test\labels")
test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))

total_correct = 0
total_gt = 0

for img_path in test_images:
    print(f"\nProcessing {img_path.name}")
    label_path = label_dir / (img_path.stem + ".txt")
    correct, gt_count = evaluate_and_classify(img_path, label_path)
    total_correct += correct
    total_gt += gt_count

if total_gt > 0:
    accuracy = (total_correct / total_gt) * 100
    print(f"\n--- Evaluation Complete ---")
    print(f"Total Correct Predictions: {total_correct}")
    print(f"Total Ground Truth Objects: {total_gt}")
    print(f"Classification Accuracy: {accuracy:.2f}%")
else:
    print("\nNo ground truth objects found to evaluate.")