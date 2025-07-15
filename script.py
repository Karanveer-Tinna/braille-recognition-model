from pathlib import Path

DATASET_FOLDER: str = r"C:\Users\veerk\OneDrive\Desktop\Braille_To_Speech\Braille Dataset\Braille Dataset"

path = Path(DATASET_FOLDER)
frequency_of_character : dict[str, int] = {chr(97+i):0 for i in range(26)}

# print(frequency_of_character)
for p in path.iterdir():
    frequency_of_character[p.name[0]] += 1

print(frequency_of_character)

label_map : dict[str, list[Path]] = {chr(97+i):[] for i in range(26)}

for p in path.iterdir():
    label_map[p.name[0]].append(p.name)

print(label_map)