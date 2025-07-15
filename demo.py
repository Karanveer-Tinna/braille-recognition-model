import cv2 as cv
import numpy as np

img : np.ndarray = cv.imread(r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Dataset\c1.JPG11dim.jpg")
height, width  = img.shape[:2]

def resize_image(image : np.ndarray, scale_factor : int | float = 10) -> np.ndarray :
    dimensions = (height*scale_factor, width*scale_factor)

    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

# cv.imshow("Image", resize_image(img))

gray : np.ndarray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("Gray", gray)

blur : np.ndarray = cv.GaussianBlur(gray, (1,1), 1)
cv.imshow("Blur", resize_image(blur))

canny : np.ndarray = cv.Canny(blur, 100, 255)
cv.imshow("Canny", resize_image(canny))

# contours : list[np.ndarray]
# contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# blank : np.ndarray = np.zeros((height, width, 3), np.uint8)
# drawn_contours : np.ndarray = cv.drawContours(blank, [cnt for cnt in contours], -1, (0, 255, 0), 1)
# cv.imshow("Drawn contours", resize_image(drawn_contours))

thresh : np.ndarray
_, thresh = cv.threshold(blur, 130, 255, cv.THRESH_BINARY_INV)
cv.imshow("Threshold", resize_image(thresh))

# _, thresh_canny = cv.threshold(canny, 130, 255, cv.THRESH_BINARY)
# cv.imshow("Threshold with canny", resize_image(thresh_canny))

cv.waitKey(0)
