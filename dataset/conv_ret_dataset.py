import os
import shutil
from PIL import Image

DATASET_PATH = r"path\to\original\dataset"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
PADDING_RATIO = 0.05

CLASS_MAP = {
    "Mild": 0,
    "Moderate": 1,
    "Proliferative_DR": 2,
    "No_DR": 3,
    "Severe": 4
}

def process_split(split_name):
    split_path = os.path.join(DATASET_PATH, split_name)
    images_out = os.path.join(split_path, "images")
    labels_out = os.path.join(split_path, "labels")

    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    for class_name in sorted(os.listdir(split_path)):
        class_path = os.path.join(split_path, class_name)

        if not os.path.isdir(class_path):
            continue
        if class_name in ["images", "labels"]:
            continue
        if class_name not in CLASS_MAP:
            continue

        class_id = CLASS_MAP[class_name]

        for file in os.listdir(class_path):
            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue

            src_img_path = os.path.join(class_path, file)
            dst_img_path = os.path.join(images_out, file)
            shutil.copy(src_img_path, dst_img_path)

            x_center = 0.5
            y_center = 0.5
            box_width = 1.0 - (2 * PADDING_RATIO)
            box_height = 1.0 - (2 * PADDING_RATIO)

            label_file = os.path.splitext(file)[0] + ".txt"
            label_path = os.path.join(labels_out, label_file)

            with open(label_path, "w") as f:
                f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

process_split("train")
process_split("valid")


