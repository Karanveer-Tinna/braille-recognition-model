import cv2 as cv
import numpy as np

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
    contours, _ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    dot_boxes = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if 10 < area < 100:  # Tune for dot size
            x, y, w, h = cv.boundingRect(cnt)
            dot_boxes.append((x, y, w, h))
    return dot_boxes

def group_into_lines(dot_boxes, y_tolerance=15):
    # Sort by Y first
    dot_boxes.sort(key=lambda b: b[1])
    lines = []
    current_line = []

    for box in dot_boxes:
        if not current_line:
            current_line.append(box)
        else:
            if abs(box[1] - current_line[-1][1]) < y_tolerance:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]
    if current_line:
        lines.append(current_line)

    return lines

def group_line_into_cells(line, x_tolerance=30):
    # Sort line left to right
    line.sort(key=lambda b: b[0])
    cells = []
    current_cell = []

    for box in line:
        if not current_cell:
            current_cell.append(box)
        else:
            if abs(box[0] - current_cell[-1][0]) < x_tolerance:
                current_cell.append(box)
            else:
                cells.append(current_cell)
                current_cell = [box]
    if current_cell:
        cells.append(current_cell)

    return cells

def group_into_braille_cells(dot_boxes, y_tolerance=20, x_tolerance=30):
    lines = group_into_lines(dot_boxes, y_tolerance)
    all_cells = []
    for line in lines:
        cells = group_line_into_cells(line, x_tolerance)
        all_cells.extend(cells)
    return all_cells

def draw_cells(img, cells):
    output = img.copy()
    for cell in cells:
        # Merge bounding boxes
        xs = [x for x, y, w, h in cell]
        ys = [y for x, y, w, h in cell]
        ws = [w for x, y, w, h in cell]
        hs = [h for x, y, w, h in cell]
        x_min = min(xs)
        y_min = min(ys)
        x_max = max([x + w for x, w in zip(xs, ws)])
        y_max = max([y + h for y, h in zip(ys, hs)])
        cv.rectangle(output, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    return output

# --- USAGE ---
img = cv.imread(r"Braille Dataset\Braille Document\datasets-braille\data\images\train\0000007_jpg.rf.cddd76cb910e100ff75766e26740a900.jpg")
thresh = preprocess_image(img)
dot_boxes = segment_braille_dots(thresh)
cells = group_into_braille_cells(dot_boxes)
result = draw_cells(img, cells)

cv.imshow("Braille Cells", result)
cv.waitKey(0)
cv.destroyAllWindows()
