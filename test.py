from keras.models import load_model
import cv2 as cv
import numpy as np

# Load the trained model
model = load_model("braille_cnn_model.h5")

# Define Braille-to-Character Mapping (adjust according to your training labels)
braille_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
    22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Load and preprocess image
img = cv.imread(r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Dataset\a1.JPG1whs.jpg")

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

def extract_cells(thresh_img: np.ndarray, contours: list) -> list:
    """Extract individual Braille cells based on bounding boxes of grouped dots"""
    cells = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        # Crop the region around the dot (expand slightly to include full cell)
        pad = 5
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(thresh_img.shape[1], x+w+pad), min(thresh_img.shape[0], y+h+pad)
        cell_img = thresh_img[y1:y2, x1:x2]
        cells.append(cell_img)
    return cells

def predict_character(cell_img: np.ndarray) -> str:
    """Resize, normalize, and predict a single Braille cell"""
    IMG_SIZE = 28  # Change based on what you trained with
    resized = cv.resize(cell_img, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)  # (batch, H, W, channels)
    pred = model.predict(reshaped)
    class_id = np.argmax(pred)
    return braille_mapping.get(class_id, '?')

# ---- Run the pipeline ----
thresh = preprocess_image(img)
dot_contours = segment_braille_dots(thresh)
cells = extract_cells(thresh, dot_contours)

predicted_text = ''.join([predict_character(cell) for cell in cells])
print("Predicted Braille Text:", predicted_text)
