import cv2
import torch
import matplotlib.pyplot as plt

from utils.inference import load_model, generate_heatmap
from utils.visualize import overlay_heatmap, draw_bounding_boxes

# CHANGE image name if needed
img_path = "data/archive/TRAINING_CG-1050\TRAINING/TAMPERED/Im1_col2.jpg"

image = cv2.imread(img_path)
if image is None:
    raise ValueError("Image not found")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("models/tamper_classifier.pth", device)

heatmap = generate_heatmap(image, model, device)
overlay = overlay_heatmap(image, heatmap)
boxed = draw_bounding_boxes(image, heatmap)

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Heatmap Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Bounding Boxes")
plt.imshow(boxed)
plt.axis("off")

plt.show()
