import cv2
import numpy as np

def preprocess_image(image):
    """
    Resize and normalize image for ResNet.
    """
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = (image - mean) / std
    return image 

def extract_patches(image, patch_size=64, stride=32):
    patches = []
    h, w, _ = image.shape
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    return np.array(patches)
