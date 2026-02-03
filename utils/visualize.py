import cv2
import numpy as np


def overlay_heatmap(image, heatmap, alpha=0.5):
    heatmap_norm = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def draw_bounding_boxes(image, heatmap, threshold=0.6, min_area=500):
    """
    Draw bounding boxes around high-probability tampered regions.
    """
    binary_map = (heatmap > threshold).astype("uint8") * 255

    contours, _ = cv2.findContours(
        binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxed_image = image.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(
            boxed_image,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    return boxed_image
