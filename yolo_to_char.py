import cv2 as cv
import numpy as np  

with open(r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Document\datasets-braille\data\labels\train\0_jpg.rf.9af84f2dce90d11fb3d643c0322c18d5.txt", "r", encoding="utf-8") as f:
    yolo_lines = f.readlines()

print(yolo_lines)

braille_map = {
    0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'j',
    10:'k', 11:'l', 12:'m', 13:'n', 14:'o', 15:'p', 16:'q', 17:'r', 18:'s', 
    19:'t', 20:'u', 21:'v', 22:'w', 23:'x', 24:'y', 25:'z'
}

labels = []
for line in yolo_lines:
    parts = line.strip().split()
    cls_id = int(parts[0])
    labels.append((cls_id, float(parts[1]), float(parts[2])))  # (class, x_center, y_center)

# ✅ Sort by top-to-bottom, left-to-right (Braille reading order)
# labels.sort(key=lambda x: (x[2], x[1]))

# ✅ Convert class IDs to letters
english_text = ''.join(braille_map.get(cls, '?') for cls, _, _ in labels)

print("Decoded Text:", english_text)
# C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Document\datasets-braille\data\images\train\0_jpg.rf.9af84f2dce90d11fb3d643c0322c18d5.jpg