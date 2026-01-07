import os
from PIL import Image

DATASET_PATH = r"path\to\original\dataset"
SPLITS = ["train", "valid"]
NUM_CLASSES = 5
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

errors = []

for split in SPLITS:
    split_path = os.path.join(DATASET_PATH, split)
    images_dir = os.path.join(split_path, "images")
    labels_dir = os.path.join(split_path, "labels")

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        errors.append(f"{split}: images/ or labels/ missing")
        continue

    images = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    for img_name in images:
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")

        if not os.path.isfile(label_path):
            errors.append(f"{split}: Missing label for {img_name}")
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        if len(lines) != 1:
            errors.append(f"{split}: {img_name} has {len(lines)} labels")
            continue

        parts = lines[0].strip().split()
        if len(parts) != 5:
            errors.append(f"{split}: Invalid label format in {img_name}")
            continue

        class_id, x, y, w, h = map(float, parts)

        if not (0 <= class_id < NUM_CLASSES):
            errors.append(f"{split}: Invalid class id in {img_name}")

        for v in [x, y, w, h]:
            if not (0.0 < v <= 1.0):
                errors.append(f"{split}: Value out of range in {img_name}")

        with Image.open(img_path) as img:
            iw, ih = img.size
            bw = w * iw
            bh = h * ih

            if bw > iw or bh > ih:
                errors.append(f"{split}: BBox exceeds image in {img_name}")

if errors:
    print("invalid")
    for e in errors:
        print("-", e)
else:
    print("valid")