import os
import random
import cv2

DATASET_PATH = r"path\to\original\dataset"
SPLIT = "train"

CLASS_NAMES = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative"
}

images_dir = os.path.join(DATASET_PATH, SPLIT, "images")
labels_dir = os.path.join(DATASET_PATH, SPLIT, "labels")

images = [
    f for f in os.listdir(images_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

random.shuffle(images)

for img_name in images:
    img_path = os.path.join(images_dir, img_name)
    label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")

    if not os.path.isfile(label_path):
        continue

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    with open(label_path, "r") as f:
        class_id, x, y, bw, bh = map(float, f.readline().split())

    cx = int(x * w)
    cy = int(y * h)
    bw = int(bw * w)
    bh = int(bh * h)

    x1 = int(cx - bw / 2)
    y1 = int(cy - bh / 2)
    x2 = int(cx + bw / 2)
    y2 = int(cy + bh / 2)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = CLASS_NAMES[int(class_id)]
    cv2.putText(
        img,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Dataset Preview", img)
    key = cv2.waitKey(0)

    if key == ord("q"):
        break

cv2.destroyAllWindows()
