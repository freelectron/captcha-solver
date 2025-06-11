import json
import os
import string

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class Captcha100kDataset(Dataset):
    allowed_chars = (
        string.ascii_letters + string.digits + "?"
    )  # added ? as an empty character (used for padding)

    def __init__(self, img_folder: str, ann_folder: str):
        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.ann_files = [f for f in os.listdir(ann_folder) if f.endswith(".json")]
        self.img_files = [f for f in os.listdir(img_folder) if f.endswith(".png")]

    @classmethod
    def _char_to_label(cls, char: str) -> int:
        return cls.allowed_chars.index(char)

    def __len__(self):
        return len(self.ann_files)

    @classmethod
    def num_classes(cls):
        return len(cls.allowed_chars)

    def _load_annotation(self, filepath: str) -> list:
        with open(filepath, "r") as file:
            ann_raw = json.load(file)

        char_labels = list()
        obj_characters = ann_raw.get("objects", [])
        for obj_ch in obj_characters:
            label = obj_ch.get("classTitle", None)
            if label is None:
                raise ValueError("Image object without label found in annotations.")
            else:
                char_labels.append(self._char_to_label(label))

        return char_labels

    @staticmethod
    def pad_to_length(tensor, target_length, pad_value=-1):
        # Only pads if tensor is shorter than target_length
        if tensor.size(0) < target_length:
            pad_amount = target_length - tensor.size(0)
            # (left, right) padding for 1D: (0, pad_amount)
            tensor = nn.functional.pad(tensor, (0, pad_amount), value=pad_value)
        return tensor

    def __getitem__(self, idx: int):
        ann_filename = self.ann_files[idx]
        img_filepath = os.path.join(
            self.img_folder, ".".join(ann_filename.split(".")[:-1])
        )
        img_arr = cv2.imread(img_filepath)
        ann_char_labels = self._load_annotation(
            os.path.join(self.ann_folder, ann_filename)
        )
        labels = torch.Tensor(ann_char_labels).long()
        # -1 because the first class id is 0, and we want to pad with the last class id
        labels_padded = (
            labels
            if len(ann_char_labels) > 5
            else self.pad_to_length(labels, 6, pad_value=self.num_classes() - 1)
        )
        img = torch.Tensor(img_arr)
        img = img.permute(2, 0, 1)

        return img, labels_padded


class Captcha100kDatasetLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    img_folder = "../../data/captcha100k/sample/img"
    ann_folder = "../../data/captcha100k/sample/ann"
    dataset = Captcha100kDataset(img_folder, ann_folder)

    print(f"Dataset size: {len(dataset)}")
    img, labels = dataset[0]
    print(f"Sample image shape: {img.shape}, Labels: {labels}")

    # from src.main import load_annotation, load_image, TrainImage
    # sample_idx = 32
    # img_path = os.path.join(img_folder, f"{sample_idx}.png")
    # image_example = img.permute(1,2,0).cpu().numpy().astype(np.int32) #load_image(img_path)
    # ann_path = os.path.join(ann_folder, f"{sample_idx}.png.json")
    # ann_example = load_annotation(ann_path)
    # img = TrainImage(image=image_example, annotations=ann_example)
    # img.show_with_bounding_boxes()

    sample_load = Captcha100kDatasetLoader(dataset, batch_size=4, shuffle=True)
    for img, labels in sample_load:
        print(f"Batch image shape: {img.shape}, Labels: {labels.shape}")
        break
