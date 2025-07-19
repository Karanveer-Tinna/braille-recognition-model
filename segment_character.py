import cv2 as cv
import numpy as np

HEIGHT, WIDTH = 0, 0

def main():
    global HEIGHT, WIDTH
    # img = cv.imread(r"Braille Dataset/Braille Document/datasets-braille/data/images/test/IMG_5466_jpg.rf.9fc8aa37446576204dca7ab136c4513e.jpg")
    img = cv.imread(r"Braille Dataset\Braille Document\datasets-braille\data\images\train\0000007_jpg.rf.cddd76cb910e100ff75766e26740a900.jpg")
    cv.imshow("Img", img)
    
    HEIGHT, WIDTH  = img.shape[:2]
    preprocessed = preprocess_image(img)
    cv.imshow("Preprocess", preprocessed)

    braille_cells = find_dot_centers(preprocessed)
    segment = img.copy()

    for (x, y, w, h) in braille_cells:
        cv.rectangle(segment, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow("Char", segment)

    cv.waitKey(0)
    cv.destroyAllWindows()


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
    # Step 1: Sort all boxes by Y (top) coordinate
    boxes_sorted = sorted(boxes, key=lambda b: b[1])
    
    rows = []
    current_row = []
    
    for box in boxes_sorted:
        x, y, w, h = box
        if not current_row:
            current_row.append(box)
        else:
            # Compare Y with the first box in the current row
            _, ref_y, _, ref_h = current_row[0]
            if abs(y - ref_y) <= y_tolerance:
                current_row.append(box)
            else:
                # Sort current row by X (left to right)
                current_row = sorted(current_row, key=lambda b: b[0])
                rows.append(current_row)
                current_row = [box]
    
    # Sort last row
    if current_row:
        current_row = sorted(current_row, key=lambda b: b[0])
        rows.append(current_row)

    # Flatten the list of rows
    return [box for row in rows for box in row]



def find_dot_centers(thresh_img: np.ndarray) -> list:
    contours, _ = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # centers = []

    boxes = [cv.boundingRect(cnt) for cnt in contours if 8 < cv.contourArea(cnt) < 100]
    boxes_sorted = sort_boxes_rowwise_with_tolerance(boxes, y_tolerance=12)
    # boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))

    boxes_2 = merge_close_boxes(boxes_sorted)
    
    boxes_3 = sort_boxes_rowwise_with_tolerance(boxes_2, y_tolerance=5)
    boxes_4 = merge_overlapping_boxes(boxes_3)

    return boxes_4

    # for cnt in contours:
    #     # area = cv.contourArea(cnt)
    #     # if 1 < area < 20:  # adjust for dot size
    #     x, y, w, h = cv.boundingRect(cnt)
    #     cx = x + w // 2
    #     cy = y + h // 2
    #     centers.append([cx, cy])
    #     cv.circle(dot_images, [cx, cy], radius=2, color=(0,255,0), thickness=-1)
    
    # return np.array(centers)

if __name__ == "__main__":
    main()