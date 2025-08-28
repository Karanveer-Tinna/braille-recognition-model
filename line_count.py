from pathlib import Path

train_lbl_dir = r"C:\Users\veerk\Downloads\braille.v2i.yolov11\train\labels"

train_directory = Path(train_lbl_dir)
char_train = 0

for file in train_directory.iterdir():
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            char_train += 1

print("Number of Characters in Train Directory :", char_train)

test_lbl_dir = r"C:\Users\veerk\Downloads\braille.v2i.yolov11\test\labels"
test_directory = Path(test_lbl_dir)
char_test = 0

for file in test_directory.iterdir():
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            char_test += 1

print("Number of Characters in Test Directory :", char_test)


valid_lbl_dir = r"C:\Users\veerk\Downloads\braille.v2i.yolov11\valid\labels"
valid_directory = Path(valid_lbl_dir)
char_valid = 0

for file in valid_directory.iterdir():
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            char_valid += 1

print("Number of Characters in Valid Directory :", char_valid)

