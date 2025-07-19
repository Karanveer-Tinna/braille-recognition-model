import cv2 as cv
import numpy as np
# from keras.models import load_model

# img = cv.imread(r"Braille Dataset\Braille Document\datasets-braille\data\images\test\IMG_4479_jpg.rf.74bfb6d141e15e92aa9aa0cb70236cd5.jpg")
img = cv.imread(r"Braille Dataset/Braille Document/datasets-braille/data/images/test/IMG_5466_jpg.rf.9fc8aa37446576204dca7ab136c4513e.jpg")
cv.imshow("Img", img)
height, width  = img.shape[:2]

# def preprocess_image(img : np.ndarray) -> np.ndarray:    
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     blurred = cv.GaussianBlur(gray, (3, 3), 1)
#     canny = cv.Canny(blurred, 100, 255)     
#     _, thresh = cv.threshold(canny, 130, 255, cv.THRESH_BINARY)
#     return blurred
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

preprocessed = preprocess_image(img)
cv.imshow("Preprocess", preprocessed)

# model = load_model("braille_cnn_model.h5")

def segment_braille(thresh_img: np.ndarray, original_img: np.ndarray) -> np.ndarray:
    contours, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    blank : np.ndarray = np.zeros((height, width, 3), np.uint8)
    drawn_contours : np.ndarray = cv.drawContours(blank, [cnt for cnt in contours], -1, (0, 255, 0), 1)
    cv.imshow("Drawn contours", drawn_contours)
    print(contours)

    result = original_img.copy()
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        if 1 < area < 100:  # Filter small noise and big unwanted areas
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return result

segmented = segment_braille(preprocessed, img)
cv.imshow("Segmented", segmented)

cv.waitKey(0)
cv.destroyAllWindows()