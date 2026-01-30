import cv2
import numpy as np

def preprocess_image(image, size=(256, 256)):
    image = cv2.resize(image, size)
    image = image.astype(np.float32) / 255.0
    return image

def extract_patches(image, patch_size=64, stride=32):
    patches = []
    h, w, _ = image.shape
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    return np.array(patches)
