import torch
import numpy as np
from utils.preprocess import preprocess_image, extract_patches
from models.patch_cnn import PatchCNN


def load_model(model_path, device):
    model = PatchCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def generate_heatmap(image, model, device, patch_size=64, stride=32):
    """
    Generates a tampering heatmap using patch-level CNN predictions.
    Skips incomplete edge patches to avoid convolution errors.
    """

    h, w, _ = image.shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    # Preprocess full image
    image_proc = preprocess_image(image)

    # Slide window over image
    for y in range(0, h, stride):
        for x in range(0, w, stride):

            patch = image_proc[y:y + patch_size, x:x + patch_size]

            # 🔑 Skip incomplete patches (VERY IMPORTANT)
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue

            patch_tensor = (
                torch.tensor(patch, dtype=torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():
                output = model(patch_tensor)
                prob = torch.softmax(output, dim=1)[0, 1].item()

            heatmap[y:y + patch_size, x:x + patch_size] += prob
            count[y:y + patch_size, x:x + patch_size] += 1

    # Normalize heatmap
    heatmap = heatmap / (count + 1e-6)
    return heatmap
