import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import sys 
sys.stderr=open(os.devnull,'w')
from utils.preprocess import preprocess_image

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


def collect_images(root_dir):
    paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(IMG_EXTENSIONS):
                paths.append(os.path.join(root, f))
    return paths


class PatchDataset(Dataset):
    """
    Memory-safe patch dataset for pretrained ResNet backbones.
    Extracts patches on-the-fly instead of storing all in RAM.
    """

    def __init__(self, root_dir, label, patch_size=224, stride=112):
        self.image_paths = collect_images(root_dir)
        self.label = label
        self.patch_size = patch_size
        self.stride = stride

        # Index map: (image_index, y, x)
        self.index_map = []
        self._build_index()

    def _build_index(self):
        for img_idx, img_path in enumerate(self.image_paths):
            image = cv2.imread(img_path)
            if image is None:
                continue

            h, w, _ = image.shape

            # 🔑 Guarantee minimum size
            if h < self.patch_size or w < self.patch_size:
                h = max(h, self.patch_size)
                w = max(w, self.patch_size)

            for y in range(0, h - self.patch_size + 1, self.stride):
                for x in range(0, w - self.patch_size + 1, self.stride):
                    self.index_map.append((img_idx, y, x))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        img_idx, y, x = self.index_map[idx]
        img_path = self.image_paths[img_idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, _ = image.shape
        if h < self.patch_size or w < self.patch_size:
            image = cv2.resize(
                image,
                (max(w, self.patch_size), max(h, self.patch_size))
            )

        patch = image[y:y + self.patch_size, x:x + self.patch_size]
        patch = preprocess_image(patch)

        patch = torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1)

        return patch, self.label
