import cv2 as cv
import numpy as np

# img : np.ndarray = cv.imread(r"C:\Users\veerk\Downloads\braille.v2i.yolov11\train\images\braille_png.rf.ed7ab2289cb97f7f605cd1545e6560bb.jpg")
# img : np.ndarray = cv.imread(r"C:\Users\veerk\Downloads\braille.v2i.yolov11\train\images\braille_png_jpg.rf.5136c89cb146bf9d55e6bb683f92892d.jpg")
img : np.ndarray = cv.imread(r"C:\Users\veerk\Downloads\braille.v2i.yolov11\train\images\braillezs_png_jpg.rf.a01d4b704b575178a1eb03abf51624f2.jpg")
# img : np.ndarray = cv.imread(r"C:\Users\veerk\Downloads\braille.v2i.yolov11\train\images\braille_png_jpg.rf.42cb693ffc346c54e7f0c6fd0e28157d.jpg")
# img : np.ndarray = cv.imread(r"C:\Users\veerk\Downloads\braille.v2i.yolov11\train\images\braille_png_jpg.rf.48e59e299750efbdbd447123cb150b68.jpg")
# img : np.ndarray = cv.imread(r"C:\Users\veerk\Downloads\braille.v2i.yolov11\train\images\braille_png_jpg.rf.25eef4787c4d1726499bff039935f7ad.jpg")


height, width  = img.shape[:2]

def resize_image(image : np.ndarray, scale_factor : int | float = 10) -> np.ndarray :
    dimensions = (height*scale_factor, width*scale_factor)

    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

equalized = clahe.apply(gray)

denoised = cv.bilateralFilter(equalized, 9, 100, 100)

thresh = cv.adaptiveThreshold(denoised, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, blockSize=5, C=2)
cv.imshow("Thresh", thresh)


cv.waitKey(0)
cv.destroyAllWindows()
