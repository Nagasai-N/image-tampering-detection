import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.preprocess import preprocess_image, extract_patches

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


def collect_images(root_dir):
    paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(IMG_EXTENSIONS):
                paths.append(os.path.join(root, f))
    return paths


class PatchDataset(Dataset):
    def __init__(self, root_dir, label, patch_size=64, stride=32):
        self.image_paths = collect_images(root_dir)
        self.label = label
        self.patch_size = patch_size
        self.stride = stride

        self.patches = []
        self.labels = []
        self._prepare_patches()

    def _prepare_patches(self):
        for img_path in self.image_paths:
            image = cv2.imread(img_path)
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = preprocess_image(image)

            patches = extract_patches(
                image,
                patch_size=self.patch_size,
                stride=self.stride
            )

            for patch in patches:
                self.patches.append(patch)
                self.labels.append(self.label)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]

        patch = torch.tensor(patch).permute(2, 0, 1)
        return patch, label
