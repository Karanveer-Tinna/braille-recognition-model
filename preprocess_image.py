import cv2 as cv
import numpy as np

# img : np.ndarray = cv.imread(r"C:\Users\veerk\Downloads\braille.v2i.yolov11\train\images\braille_png.rf.ed7ab2289cb97f7f605cd1545e6560bb.jpg")
# img : np.ndarray = cv.imread(r"C:\Users\veerk\Downloads\braille.v2i.yolov11\train\images\braille_png_jpg.rf.5136c89cb146bf9d55e6bb683f92892d.jpg")
img : np.ndarray = cv.imread(r"C:\Users\veerk\Downloads\braille.v2i.yolov11\train\images\db11_png_jpg.rf.fd33a84c53919e9a3c95c32ced2dc7c1.jpg")
# img : np.ndarray = cv.imread(r"C:\Users\veerk\Downloads\braille.v2i.yolov11\train\images\braille_png_jpg.rf.42cb693ffc346c54e7f0c6fd0e28157d.jpg")
# img : np.ndarray = cv.imread(r"C:\Users\veerk\Downloads\braille.v2i.yolov11\train\images\braille_png_jpg.rf.48e59e299750efbdbd447123cb150b68.jpg")
# img : np.ndarray = cv.imread(r"C:\Users\veerk\Downloads\braille.v2i.yolov11\train\images\braille_png_jpg.rf.25eef4787c4d1726499bff039935f7ad.jpg")

cv.imshow("Original", img)
height, width  = img.shape[:2]

def resize_image(image : np.ndarray, scale_factor : int | float = 10) -> np.ndarray :
    dimensions = (height*scale_factor, width*scale_factor)

    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))

equalized = clahe.apply(gray)
blurred = cv.GaussianBlur(equalized, (5,5), 0)

denoised = cv.bilateralFilter(blurred, 9, 100, 100)

thresh = cv.adaptiveThreshold(denoised, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, blockSize=5, C=2)
cv.imshow("Thresh", thresh)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
opened = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
cv.imshow("New + blur", opened)

denoised = cv.bilateralFilter(equalized, 9, 100, 100)

thresh = cv.adaptiveThreshold(denoised, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, blockSize=5, C=2)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
opened = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
cv.imshow("New", opened)

cv.waitKey(0)
cv.destroyAllWindows()
