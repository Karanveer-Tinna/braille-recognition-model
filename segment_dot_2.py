import cv2 as cv
import numpy as np
# from keras.models import load_model
from sklearn.cluster import DBSCAN

img = cv.imread(r"Braille Dataset/Braille Document/datasets-braille/data/images/test/IMG_5466_jpg.rf.9fc8aa37446576204dca7ab136c4513e.jpg")
cv.imshow("Img", img)
height, width  = img.shape[:2]

def preprocess_image(img: np.ndarray) -> np.ndarray:
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # Denoise with bilateral filter (preserves edges better than Gaussian blur)
    denoised = cv.bilateralFilter(equalized, d=9, sigmaColor=200, sigmaSpace=200)

    # Adaptive Thresholding (better in uneven lighting conditions)
    thresh = cv.adaptiveThreshold(
        denoised, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,
        blockSize=5, C=2
    )

    return thresh

def segment_braille_dots(thresh_img: np.ndarray) -> list:
    # Find contours
    contours, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Filter based on area
    dot_contours = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if 10 < area < 100:  # These values may need tuning
            dot_contours.append(cnt)

    return dot_contours

def draw_dots(img: np.ndarray, contours: list) -> np.ndarray:
    output = img.copy()
    for cnt in contours:
        (x, y, w, h) = cv.boundingRect(cnt)
        cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return output

thresh = preprocess_image(img)
dot_contours = segment_braille_dots(thresh)
result = draw_dots(img, dot_contours)

cv.imshow("Braille Dots", result)
cv.waitKey(0)
cv.destroyAllWindows()