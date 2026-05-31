from keras.models import load_model
import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN

braille_mapping = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
        8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
        15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
        22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}
model = load_model("braille_cnn_model.h5")
print(model.input_shape)

def main():
    img = cv.imread(r"Braille Dataset\Braille Document\datasets-braille\data\images\test\IMG_5231_jpg.rf.f2edf09192c7c9b39d295a41c44519fa.jpg")

    thresh = preprocess_image(img)
    # cv.imshow("Preprocessed image", thresh)

    dot_contours = segment_braille_dots(thresh)
    # blank : np.ndarray = np.zeros(img.shape, np.uint8)
    # drawn_contours : np.ndarray = cv.drawContours(blank, [cnt for cnt in dot_contours], -1, (0, 255, 0), 1)
    # cv.imshow("Drawn Contours", drawn_contours)
    # cv.waitKey(0)

    cell_groups = group_dot_contours_into_cells(dot_contours)
    cells, cell_boxes = extract_cells(thresh, cell_groups)

    # Draw cell boxes
    blank = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)
    for (x1, y1, x2, y2) in cell_boxes:
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv.imshow("Braille Cells", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    predicted_text = ''.join([predict_character(cell) for cell in cells])
    print("Predicted Braille Text:", predicted_text)


def preprocess_image(img: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    denoised = cv.bilateralFilter(equalized, d=9, sigmaColor=200, sigmaSpace=200)
    thresh = cv.adaptiveThreshold(
        denoised, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,
        blockSize=5, C=2
    )
    return thresh

def segment_braille_dots(thresh_img: np.ndarray) -> list:
    contours, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    dot_contours = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if 10 < area < 100:
            dot_contours.append(cnt)
    return dot_contours

def extract_cells(thresh_img: np.ndarray, cell_groups: list) -> list:
    """Extract Braille cells by grouping dots into cell regions"""
    cells = []
    boxes = []

    for group in cell_groups:
        # Combine all contours in the group
        x1, y1, x2, y2 = np.inf, np.inf, -np.inf, -np.inf
        for cnt in group:
            x, y, w, h = cv.boundingRect(cnt)
            x1, y1 = min(x1, x), min(y1, y)
            x2, y2 = max(x2, x+w), max(y2, y+h)
        
        pad = 5
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(thresh_img.shape[1], x2 + pad), min(thresh_img.shape[0], y2 + pad)

        cell_img = thresh_img[y1:y2, x1:x2]
        cells.append(cell_img)
        boxes.append((x1, y1, x2, y2))  # save box for visualization

    return cells, boxes

def group_dot_contours_into_cells(contours, eps=30, min_samples=1):
    """Groups nearby dot contours into clusters (cells) using DBSCAN"""
    centers = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        centers.append([x + w//2, y + h//2]) 

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
    
    groups = {}
    for idx, label in enumerate(clustering.labels_):
        if label not in groups:
            groups[label] = []
        groups[label].append(contours[idx])

    return list(groups.values())

def predict_character(cell_img: np.ndarray) -> str:
    """Resize, normalize, and predict a single Braille cell"""
    IMG_SIZE = 28  # Change based on what you trained with
    resized = cv.resize(cell_img, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)  # (batch, H, W, channels)
    pred = model.predict(reshaped)
    class_id = np.argmax(pred)
    return braille_mapping.get(class_id, '?')

if __name__ == "__main__":
    main()