from dataclasses import dataclass
import json
import logging
import os

import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.optim
from torch import nn

from src.model.crnn import CRNN
from src.data_loading.loader import Captcha100kDatasetLoader, Captcha100kDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    # IMG_FOLDER = "../data/captcha100k/sample/img/"
    # ANN_FOLDER = "../data/captcha100k/sample/ann/"
    # sample_idx = 0
    # img_path = os.path.join(IMG_FOLDER, f"{sample_idx}.png")
    # image_example = load_image(img_path)
    # ann_path = os.path.join(ANN_FOLDER, f"{sample_idx}.png.json")
    # ann_example = load_annotation(ann_path)
    # img = TrainImage(image=image_example, annotations=ann_example)
    # img.show_with_bounding_boxes()

    img_folder_train = "../data/captcha100k/train/img/"
    ann_folder_train = "../data/captcha100k/train/ann/"
    dataset_train = Captcha100kDataset(img_folder_train, ann_folder_train)
    train_dataloader = Captcha100kDatasetLoader(dataset_train, batch_size=4)

    img_folder_validation = "../data/captcha100k/validation/img/"
    ann_folder_validation = "../data/captcha100k/validation/ann/"
    dataset_validation = Captcha100kDataset(img_folder_train, ann_folder_train)
    validation_dataloader = Captcha100kDatasetLoader(dataset_validation, batch_size=4)

    img_folder_test = "../data/captcha100k/test/img/"
    ann_folder_test = "../data/captcha100k/test/ann/"
    dataset_test = Captcha100kDataset(img_folder_train, ann_folder_train)
    test_dataloader = Captcha100kDatasetLoader(dataset_test, batch_size=4)

    model = CRNN(img_h=60, img_w=160, in_dimensions_rnn=265, hidden_dimensions_rnn=128, num_classes=Captcha100kDataset.num_classes())
    device = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # Enable MPS fallback for compatibility: a few things are not implemented in MPS yet
    model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # last class is the blank label for CTC loss, so we use num_classes()-1
    criterion = nn.CTCLoss(blank=Captcha100kDataset.num_classes()-1)

    for epoch in range(10):
        train_loss = 0
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False):
            images = images.to(device=device)
            labels = labels.to(device=device)

            optimizer.zero_grad()
            preds = model(images)
            preds = preds.permute(1, 0, 2)  # Change shape to (seq_len, batch_size, num_classes) for CTC loss
            preds_log_probs = nn.functional.log_softmax(preds, dim=2)

            pred_lengths = torch.full(size=(preds.size(1), ), fill_value=preds_log_probs.size(0), dtype=torch.int32)
            labels_lengths = torch.full(size=(labels.size(0), ), fill_value=labels.size(1), dtype=torch.int32)
            loss = criterion(preds_log_probs.cpu(), labels.cpu(), pred_lengths.cpu(), labels_lengths.cpu())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        logger.info("After epoch: {} | loss: {:.4f} | train loss: {:.4f}".format(epoch + 1, train_loss, avg_train_loss))
