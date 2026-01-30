import cv2
import matplotlib.pyplot as plt
from utils.preprocess import preprocess_image, extract_patches

# CHANGE THIS to any real image filename from your dataset
img_path = "data/archive/TRAINING_CG-1050\TRAINING/ORIGINAL/Im1_2_col.jpg"

# Load image
image = cv2.imread(img_path)
if image is None:
    raise ValueError("Image not found or path incorrect")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Preprocess
image = preprocess_image(image)

# Extract patches
patches = extract_patches(image)

print("Total patches extracted:", len(patches))
print("Patch shape:", patches[0].shape)

# Visualize first 5 patches
plt.figure(figsize=(10, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(patches[i])
    plt.title(f"Patch {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()
