from dataclasses import dataclass
import json
import logging
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_FOLDER = "../data/captcha100k/sample/img/"
ANN_FOLDER = "../data/captcha100k/sample/ann/"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class TrainImage:
    image: np.ndarray
    annotations: dict
    bbox_geometry_type = "rectangle"

    def __post_init__(self):
        self.bounding_boxes_labels = self.get_bounding_boxes_labels()

    def get_bounding_boxes_labels(self) -> list:
        obj_characters = self.annotations.get('objects', [])
        boxes = []
        if len(obj_characters) > 0:
            for obj_ch in obj_characters:
                label = obj_ch.get('classTitle', None)
                if label is None:
                    logger.warning("Image object without label found in annotations.")
                if obj_ch["geometryType"] == self.bbox_geometry_type:
                    points = obj_ch.get('points', [])
                    if exterior := points.get('exterior'):
                        x1, y1 = exterior[0]
                        x2, y2 = exterior[1]
                        boxes.append({
                            'x1': int(x1),
                            'y1': int(y1),
                            'x2': int(x2),
                            'y2': int(y2),
                            'label':label
                        })
        return boxes

    def show(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(self.image)
        plt.show()

    def show_with_bounding_boxes(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(self.image)
        for box in self.bounding_boxes_labels:
            rect = plt.Rectangle((box['x1'], box['y1']),
                                 box['x2'] - box['x1'],
                                 box['y2'] - box['y1'],
                                 linewidth=2,
                                 edgecolor='r',
                                 facecolor='none')
            ax.add_patch(rect)
            ax.text(box['x1'], box['y1'], box['label'], color='green', fontsize=20)
        plt.show()

def load_image(path: str) -> np.ndarray:
    return cv2.imread(path)

def load_annotation(path: str, index: int = None) -> dict:
    with open(path, 'r') as file:
        ann = json.load(file)
        ann["index"] = index if index else path.split("/")[-1].split(".")[0]
        return ann

if __name__=="__main__":
    # Load an example image
    sample_idx = 0
    img_path = os.path.join(IMG_FOLDER, f"{sample_idx}.png")
    image_example = load_image(img_path)
    ann_path = os.path.join(ANN_FOLDER, f"{sample_idx}.png.json")
    ann_example = load_annotation(ann_path)

    img = TrainImage(image=image_example, annotations=ann_example)
    #img.show_with_bounding_boxes()

