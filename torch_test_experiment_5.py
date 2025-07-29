import cv2 as cv
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms


DATASET_DIR = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Dataset"

def main():
    X, y = [], []
    label_map = {chr(97 + i): i for i in range(26)} 

    for img_path in Path(DATASET_DIR).iterdir():
        processed = preprocess_image(cv.imread(img_path))
        X.append(processed)
        y.append(label_map[img_path.name[0]])

    X = np.array(X).astype(np.float32).reshape(-1, 1, 28, 28) / 255.0 
    y = np.array(y).astype(np.int64)

    train_loader, test_loader = split_into_loaders(X, y)
    model, device = train_model(train_loader)

    evaluate_model(test_loader, model, device)

    image_path = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Document\datasets-braille\data\images\test\IMG_4479_jpg.rf.74bfb6d141e15e92aa9aa0cb70236cd5.jpg"
    label_path = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Document\datasets-braille\data\labels\test\IMG_4479_jpg.rf.74bfb6d141e15e92aa9aa0cb70236cd5.txt"

    predicted = predict_braille_text_from_image(image_path, model, device)

    actual = get_actual_text_from_yolo_labels(label_path)
    compare_predicted_and_actual(predicted, actual)    

def preprocess_image(img: np.ndarray) -> np.ndarray:
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # Denoise with bilateral filter (preserves edges better than Gaussian blur)
    denoised = cv.bilateralFilter(equalized, d=9, sigmaColor=100, sigmaSpace=100)

    # Adaptive Thresholding (better in uneven lighting conditions)
    thresh = cv.adaptiveThreshold(
        denoised, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,
        blockSize=5, C=2
    )

    return thresh

def split_into_loaders(X: np.array, y: np.array):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

class BrailleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, 3),  # input channels = 1
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),  # 28x28 -> conv & pool -> 5x5
            nn.ReLU(),
            nn.Linear(128, 26)  # 26 output classes
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

def train_model(train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrailleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 16
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/total:.4f} | Accuracy: {100 * correct / total:.2f}%")

    return model, device

def evaluate_model(test_loader, model, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), "braille_cnn_model.pth")
    print("Model saved successfully!")

def predict_braille_text_from_image(img_path, model, device):
    img = cv.imread(img_path)
    thresh = preprocess_image(img)
    # dot_contours = segment_braille_dots(thresh)

    cell_groups = find_cell_groups(thresh)
    cells, _ = extract_cells(thresh, cell_groups)

    predicted_text = ''
    model.eval()

    for idx, cell in enumerate(cells):
        cv.imshow(f"Cell {idx}", cell)
        cv.waitKey(0)
        cv.destroyAllWindows()

    for cell in cells:
        resized = cv.resize(cell, (28, 28)).astype(np.float32) / 255.0
        tensor = torch.tensor(resized).unsqueeze(0).unsqueeze(0).to(device)  # shape: (1, 1, 28, 28)

        with torch.no_grad():
            output = model(tensor)
            pred_class = torch.argmax(output, dim=1).item()
            predicted_text += chr(97 + pred_class)

    return predicted_text

# def segment_braille_dots(thresh_img: np.ndarray) -> list:
#     contours, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     dot_contours = []
#     for cnt in contours:
#         area = cv.contourArea(cnt)
#         if 10 < area < 100:
#             dot_contours.append(cnt)
#     return dot_contours

def find_cell_groups(thresh_img: np.ndarray) -> list:
    contours, _ = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # centers = []

    boxes = [cv.boundingRect(cnt) for cnt in contours if 6 < cv.contourArea(cnt) < 100]
    boxes_sorted = sort_boxes_rowwise_with_tolerance(boxes, y_tolerance=12)

    boxes_2 = merge_close_boxes(boxes_sorted)
    
    boxes_3 = sort_boxes_rowwise_with_tolerance(boxes_2, y_tolerance=5)
    boxes_4 = merge_overlapping_boxes(boxes_3)

    return sort_boxes_rowwise_with_tolerance(boxes_4, y_tolerance = 15)



def merge_close_boxes(boxes):
    """
    Merge nearby/overlapping bounding boxes into larger boxes (Braille cells).
    max_distance controls how close boxes need to be to merge.
    """
    merged = []
    while boxes:
        x, y, w, h = boxes.pop(0)
        box1 = [x, y, x + w, y + h]

        to_merge = []
        for b in boxes:
            x2, y2, w2, h2 = b
            box2 = [x2, y2, x2 + w2, y2 + h2]

            # Check if boxes overlap or are close enough to be in the same Braille cell
            if not (box1[2] + 1*w < box2[0] or
                    box1[0] - 0.75*w> box2[2] or
                    box1[3] + 3*h < box2[1] or
                    box1[1] > box2[3]):
                to_merge.append(b)

        # Merge all nearby boxes
        for b in to_merge:
            boxes.remove(b)
            x2, y2, w2, h2 = b
            box1[0] = min(box1[0], x2)
            box1[1] = min(box1[1], y2)
            box1[2] = max(box1[2], x2 + w2)
            box1[3] = max(box1[3], y2 + h2)

        merged.append((box1[0], box1[1], box1[2] - box1[0], box1[3] - box1[1]))

    return merged

def merge_overlapping_boxes(boxes):
    """
    Merge nearby/overlapping bounding boxes into larger boxes (Braille cells).
    max_distance controls how close boxes need to be to merge.
    """
    merged = []
    while boxes:
        x, y, w, h = boxes.pop(0)
        box1 = [x, y, x + w, y + h]

        to_merge = []
        for b in boxes:
            x2, y2, w2, h2 = b
            box2 = [x2, y2, x2 + w2, y2 + h2]

            # Check if boxes overlap or are close enough to be in the same Braille cell
            if not (box1[2] < box2[0] or
                    box1[0] > box2[2] or
                    box1[3] < box2[1] or
                    box1[1] > box2[3]):
                to_merge.append(b)

        # Merge all nearby boxes
        for b in to_merge:
            boxes.remove(b)
            x2, y2, w2, h2 = b
            box1[0] = min(box1[0], x2)
            box1[1] = min(box1[1], y2)
            box1[2] = max(box1[2], x2 + w2)
            box1[3] = max(box1[3], y2 + h2)

        merged.append((box1[0], box1[1], box1[2] - box1[0], box1[3] - box1[1]))

    return merged


def sort_boxes_rowwise_with_tolerance(boxes, y_tolerance=10):
    """
    Sorts bounding boxes top to bottom, left to right within each row.
    A row is defined as a group of boxes whose top Y values are within y_tolerance.
    
    Args:
        boxes: List of tuples (x, y, w, h)
        y_tolerance: Tolerance for grouping boxes into the same row
    
    Returns:
        A list of boxes sorted row-wise.
    """
    # Step 1: Sort by Y (top coordinate)
    boxes_sorted_by_y = sorted(boxes, key=lambda b: b[1])
    
    rows = []
    current_row = []
    
    for box in boxes_sorted_by_y:
        x, y, w, h = box
        if not current_row:
            current_row.append(box)
        else:
            _, ref_y, _, ref_h = current_row[0]
            if abs(y - ref_y) <= y_tolerance:
                current_row.append(box)
            else:
                # Finalize and sort current row by X
                current_row.sort(key=lambda b: b[0])
                rows.append(current_row)
                current_row = [box]
    
    # Add the last row
    if current_row:
        current_row.sort(key=lambda b: b[0])
        rows.append(current_row)
    
    # Flatten rows into a single list
    return [box for row in rows for box in row]


def extract_cells(thresh_img: np.ndarray, cell_groups: list) -> tuple:
    cells, boxes = [], []

    for x, y, w, h in cell_groups:
        pad = 5
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(thresh_img.shape[1], x + w + pad), min(thresh_img.shape[0], y + h + pad)

        cell_img = thresh_img[y1:y2, x1:x2]
        cells.append(cell_img)
        boxes.append((x1, y1, x2, y2))

    return cells, boxes

def get_actual_text_from_yolo_labels(label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        yolo_lines = f.readlines()

    labels = []
    for line in yolo_lines:
        parts = line.strip().split()
        cls_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        labels.append((cls_id, x_center, y_center))


    actual_text = ''.join(chr(97 + cls_id) for cls_id, _, _ in labels)
    return actual_text


def compare_predicted_and_actual(predicted_text, actual_text):
    min_len = min(len(predicted_text), len(actual_text))
    matches = sum(predicted_text[i] == actual_text[i] for i in range(min_len))
    accuracy = matches / max(len(actual_text), 1) * 100  # avoid divide-by-zero
    print(f"\nActual Text:    {actual_text}")
    print(f"Predicted Text: {predicted_text}")
    print(f"Character-level Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    main()
