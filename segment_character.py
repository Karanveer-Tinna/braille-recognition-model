import cv2 as cv
import numpy as np
# from keras.models import load_model
from sklearn.cluster import DBSCAN

img = cv.imread(r"Braille Dataset/Braille Document/datasets-braille/data/images/test/IMG_5466_jpg.rf.9fc8aa37446576204dca7ab136c4513e.jpg")
cv.imshow("Img", img)
height, width  = img.shape[:2]

# def preprocess_image(img : np.ndarray) -> np.ndarray:    
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     blurred = cv.GaussianBlur(gray, (5, 5), 1)
#     canny = cv.Canny(blurred, 100, 200)     
#     _, thresh = cv.threshold(canny, 50, 255, cv.THRESH_BINARY)
#     return thresh

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

dot_images = img.copy()

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
                    box1[0] - 0.75* w> box2[2] or
                    box1[3] + 2.5*h < box2[1] or
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

    boxes = [cv.boundingRect(cnt) for cnt in contours if 3 < cv.contourArea(cnt) < 100]
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

def dbscan_with_max_points(dot_centers, eps=35, min_samples=2, max_points=6):
    """
    Runs DBSCAN and limits the max number of points per cluster.
    Extra points are marked as noise (-1).
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(dot_centers)
    labels = clustering.labels_

    for label in set(labels):
        if label == -1:
            continue  # ignore noise

        # Find indices of points in this cluster
        cluster_indices = np.where(labels == label)[0]

        if len(cluster_indices) > max_points:
            # Keep only the first 'max_points', others become noise (-1)
            for idx in cluster_indices[max_points:]:
                labels[idx] = -1

    return labels

# Step 3: Cluster dots into cells
def cluster_into_cells(dot_centers: np.ndarray, original_img: np.ndarray) -> np.ndarray:
    result = original_img.copy()

    if len(dot_centers) == 0:
        return result

    # clustering = DBSCAN(eps=35, min_samples=2).fit(dot_centers)  # eps ~ distance between dots in a cell
    # labels = clustering.labels_
    labels = dbscan_with_max_points(dot_centers=dot_centers, eps=23, min_samples=1)

    for label in set(labels):
        if label == -1:
            continue  # noise
        points = dot_centers[labels == label]
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        cv.rectangle(result, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), 2)

    return result

braille_cells = find_dot_centers(preprocessed)
segment = img.copy()
# cv.imshow("Dot", dot_images)
# segmented_cells = cluster_into_cells(dot_centers, img)
# cv.imshow("Braille Cells", segmented_cells)

# print(braille_cells)
for (x, y, w, h) in braille_cells:
    cv.rectangle(segment, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv.imshow("Char", segment)

cv.waitKey(0)
cv.destroyAllWindows()