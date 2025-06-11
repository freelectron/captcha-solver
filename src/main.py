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

from src.model.crnn import CRNN, CNN_BASIC
from src.data_loading.loader import Captcha100kDatasetLoader, Captcha100kDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fomatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s | %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(fomatter)
logger.addHandler(handler)


@dataclass
class TrainImage:
    image: np.ndarray
    annotations: dict
    bbox_geometry_type = "rectangle"

    def __post_init__(self):
        self.bounding_boxes_labels = self.get_bounding_boxes_labels()

    def get_bounding_boxes_labels(self) -> list:
        obj_characters = self.annotations.get("objects", [])
        boxes = []
        if len(obj_characters) > 0:
            for obj_ch in obj_characters:
                label = obj_ch.get("classTitle", None)
                if label is None:
                    logger.warning("Image object without label found in annotations.")
                if obj_ch["geometryType"] == self.bbox_geometry_type:
                    points = obj_ch.get("points", [])
                    if exterior := points.get("exterior"):
                        x1, y1 = exterior[0]
                        x2, y2 = exterior[1]
                        boxes.append(
                            {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                                "label": label,
                            }
                        )
        return boxes

    def show(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(self.image)
        plt.show()

    def show_with_bounding_boxes(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(self.image)
        for box in self.bounding_boxes_labels:
            rect = plt.Rectangle(
                (box["x1"], box["y1"]),
                box["x2"] - box["x1"],
                box["y2"] - box["y1"],
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(box["x1"], box["y1"], box["label"], color="green", fontsize=20)
        plt.show()


def load_image(path: str) -> np.ndarray:
    return cv2.imread(path)


def load_annotation(path: str, index: int = None) -> dict:
    with open(path, "r") as file:
        ann = json.load(file)
        ann["index"] = index if index else path.split("/")[-1].split(".")[0]
        return ann

def ctc_greedy_decoder(preds, blank_class=64):
    pred_indices = torch.argmax(preds, dim=2)
    # pred_indices = pred_indices.transpose(0, 1)
    # preds_indices = pred_indices.view(pred_indices.size(0), -1).cpu().numpy().tolist()
    pred_strings = []
    for pred in pred_indices:
        collapsed = []
        prev = None
        for p in pred:
            if p != prev and p != blank_class:  # skip blanks
                collapsed.append(p.item())
            prev = p
        pred_strings.append(collapsed)

    # Make sequences to length 6
    result = list()
    for pred in pred_indices:
        if len(pred) < 6:
            right_padding = 6 - len(pred)
            pred = torch.nn.functional.pad(torch.Tensor(pred_strings), (0,right_padding))
        else:
            pred = pred[:6]
        result.append(pred)

    return torch.stack(result)


def calc_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    # Method 1: compare the first six sequences since this is where the labels should be
    # preds_classes = preds.argmax(dim=2, keepdim=True)
    # correct = preds_classes.view(preds_classes.size(0),-1)[:, :6].eq(labels).int()
    # Method 2: use CTC greedy decoder
    preds_clases = ctc_greedy_decoder(preds)
    return (sum(sum(preds_clases.eq(labels).int())).cpu().detach() / (labels.size(0) * labels.size(1))).cpu().detach()

def load_model_weights(model: CRNN, state_dict_path: str):
    if os.path.exists(state_dict_path):
        logger.info(f"Loading model weights from {state_dict_path}.")
        model.load_state_dict(torch.load(state_dict_path))
    else:
        raise FileNotFoundError(f"Model weights file {state_dict_path} does not exist.")

if __name__ == "__main__":
    # IMG_FOLDER = "../data/captcha100k/sample/img/"
    # ANN_FOLDER = "../data/captcha100k/sample/ann/"
    # sample_idx = 0
    # img_path = os.path.join(IMG_FOLDER, f"{sample_idx}.png")
    # image_example = load_image(img_path)
    # ann_path = os.path.join(ANN_FOLDER, f"{sample_idx}.png.json")
    # ann_example = load_annotation(ann_path)
    # img = TrainImage(image=image_example, annotations=ann_example)
    # img.show_with_bounding_boxes()

    BATCH_SIZE = 64
    MODEL_PARAMS = {
        "batch_size": BATCH_SIZE,
        "model_CRNN": CNN_BASIC,
    }
    START_LEARNING_RATE = 0.0001
    NUM_EPOCHS = 200

    img_folder_train = "../data/captcha100k/train/img/"
    ann_folder_train = "../data/captcha100k/train/ann/"
    dataset_train = Captcha100kDataset(img_folder_train, ann_folder_train)
    train_dataloader = Captcha100kDatasetLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=Captcha100kDataset.collate_fn
    )

    img_folder_validation = "../data/captcha100k/validation/img/"
    ann_folder_validation = "../data/captcha100k/validation/ann/"
    dataset_validation = Captcha100kDataset(img_folder_train, ann_folder_train)
    validation_dataloader = Captcha100kDatasetLoader(
        dataset_validation, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=Captcha100kDataset.collate_fn
    )

    img_folder_test = "../data/captcha100k/test/img/"
    ann_folder_test = "../data/captcha100k/test/ann/"
    dataset_test = Captcha100kDataset(img_folder_train, ann_folder_train)
    test_dataloader = Captcha100kDatasetLoader(
        dataset_test, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=Captcha100kDataset.collate_fn
    )

    model = CRNN(
        img_h=60,
        img_w=160,
        in_dimensions_rnn=265,
        hidden_dimensions_rnn=128,
        num_classes=Captcha100kDataset.num_classes(),
    )
    # load_model_weights(
    #     model,
    #     "../data/models/44abb492072ad870af24d9ad25ea11b640038a07/2025-06-11T13:25:57.522199/crnn_model.pth"
    # )

    device = "mps"
    # Enable MPS fallback for compatibility: a few things are not implemented in MPS yet
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    model.to(device=device)

    lr = START_LEARNING_RATE
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    schedular = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=NUM_EPOCHS)
    # last class is the blank label for CTC loss, so we use num_classes()-1
    criterion = nn.CTCLoss(blank=Captcha100kDataset.num_classes() - 1)

    avg_train_loses = list()
    median_train_loses = list()
    avg_train_accuracies = list()

    avg_validation_loses = list()
    median_validation_loses = list()
    avg_validation_accuracies = list()

    for epoch in range(NUM_EPOCHS):
        logger.info(f"Started training epoch {epoch+1}.")
        epoch_train_losses = list()
        for images, labels, labels_lengths in tqdm(
            train_dataloader, desc=f"Training epoch {epoch+1}", leave=False
        ):
            images = images.to(device=device)
            labels = labels.to(device=device)

            optimizer.zero_grad()
            model.train()
            preds = model(images)
            preds = preds.permute(
                1, 0, 2
            )  # Change shape to (seq_len, batch_size, num_classes) for CTC loss
            preds_log_probs = nn.functional.log_softmax(preds, dim=2)
            pred_lengths = torch.full(
                size=(preds.size(1),),
                fill_value=preds_log_probs.size(0),
                dtype=torch.int32,
            )
            # labels_lengths = torch.full(
            #     size=(labels.size(0),), fill_value=labels.size(1), dtype=torch.int32
            # )
            loss = criterion(
                preds_log_probs.cpu(),
                labels.cpu(),
                pred_lengths.cpu(),
                labels_lengths.cpu(),
            )
            loss.backward()
            optimizer.step()

            # Double check but by default: loss = the mean loss over the batch
            epoch_train_losses.append(loss.item())

        epoch_total_train_loss = sum(epoch_train_losses)
        epoch_avg_train_loss = epoch_total_train_loss / len(train_dataloader)
        epoch_median_train_loss = sorted(epoch_train_losses)[
            len(epoch_train_losses) // 2
        ]
        logger.info(
            "After epoch: {} | loss: {:.4f} | avg loss: {:.4f} | med loss: {:.4f}".format(
                epoch + 1,
                epoch_total_train_loss,
                epoch_avg_train_loss,
                epoch_median_train_loss,
            )
        )
        avg_train_loses.append(epoch_avg_train_loss)
        median_train_loses.append(epoch_median_train_loss)

        with torch.no_grad():
            epoch_validation_losses = list()
            epoch_validation_accuracies = list()
            for images, labels, _ in tqdm(
                validation_dataloader, desc="Validation", leave=False
            ):
                images = images.to(device=device)
                labels = labels.to(device=device)

                model.eval()
                preds = model(images)
                preds = preds.permute(1, 0, 2)
                preds_log_probs = nn.functional.log_softmax(preds, dim=2)
                pred_lengths = torch.full(
                    size=(preds.size(1),),
                    fill_value=preds_log_probs.size(0),
                    dtype=torch.int32,
                )
                labels_lengths = torch.full(
                    size=(labels.size(0),), fill_value=labels.size(1), dtype=torch.int32
                )
                loss = criterion(
                    preds_log_probs.cpu(),
                    labels.cpu(),
                    pred_lengths.cpu(),
                    labels_lengths.cpu(),
                )

                epoch_validation_losses.append(loss.item())
                epoch_validation_accuracies.append(
                    0
                    # float(calc_accuracy(
                    #     preds_log_probs.view(preds_log_probs.size(1),preds_log_probs.size(0),preds_log_probs.size(2)),
                    #     labels
                    # ))
                )

            epoch_total_validation_loss = sum(epoch_validation_losses)
            epoch_avg_validation_loss = epoch_total_validation_loss / len(
                validation_dataloader
            )
            epoch_median_validation_loss = sorted(epoch_validation_losses)[
                len(epoch_validation_losses) // 2
            ]
            epoch_avg_validation_accuracy = sum(epoch_validation_accuracies) / len(epoch_validation_accuracies)
            logger.info(
                "Validation after epoch: {} | loss: {:.4f} | avg loss: {:.4f} | med loss: {:.4f} | Acc {:.4f}".format(
                    epoch + 1,
                    epoch_total_validation_loss,
                    epoch_avg_validation_loss,
                    epoch_median_validation_loss,
                    epoch_avg_validation_accuracy,
                )
            )
            avg_validation_loses.append(epoch_avg_validation_loss)
            median_validation_loses.append(epoch_median_validation_loss)
            avg_validation_accuracies.append(epoch_avg_validation_accuracy)

        if epoch != 0 and epoch % 10 == 0:
            model.save_status(
                "../data/",
                MODEL_PARAMS,
            )
            logger.info(f"Model saved after epoch {epoch+1}.")