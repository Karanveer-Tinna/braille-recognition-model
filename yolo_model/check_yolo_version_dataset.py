from pathlib import Path
import yaml

yaml_path = Path("braille_data.yaml")

with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

if isinstance(data.get("names"), dict):
    print("Likely YOLOv8 format (names as dict).")
elif isinstance(data.get("names"), list):
    if "nc" in data:
        print("Likely YOLOv11 format (names as list + nc present).")
    else:
        print("Could be YOLOv8 or YOLOv11 (names as list).")
else:
    print("Unknown dataset format.")
