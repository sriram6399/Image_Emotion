import os
import json
import cv2
import numpy as np

def process_subfolders(base_folder):
    for root, dirs, files in os.walk(base_folder):
        if "metadata.json" in files:
            # Load metadata.json
            metadata_file = os.path.join(root, "metadata.json")
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            # Extract image_id and paths
            image_id = metadata.get("image_id")  # Extract image_id from metadata.json
            heatmap_path = os.path.join(root, metadata.get("sd_heatmap", ""))
            heatmap_opp_path = os.path.join(root, metadata.get("sd_heatmap_opp", ""))
            image_path = os.path.join(root, metadata.get("sd_generated_image_path", ""))
            image_opp_path = os.path.join(root, metadata.get("sd_generated_image_path_opp", ""))

            # Check if all required files exist
            if not all([os.path.exists(p) for p in [heatmap_path, heatmap_opp_path, image_path, image_opp_path]]):
                print(f"Missing files in {root}. Skipping...")
                continue

            # Process heatmap and image pairs
            draw_and_save_contours(heatmap_path, image_path, os.path.join(root, f"{image_id}_contour_boxes.png"))
            draw_and_save_contours(heatmap_opp_path, image_opp_path, os.path.join(root, f"{image_id}_contour_boxes_opp.png"))

def draw_and_save_contours(heatmap_path, image_path, save_path):
    # Load heatmap and image
    heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)

    # Compute percentile-based threshold (e.g., 95th percentile)
    threshold_value = np.percentile(heatmap, 95)  # Top 5% brightest pixels
    print(f"Using threshold: {threshold_value}")

    # Apply binary thresholding
    _, binary_heatmap = cv2.threshold(heatmap, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the image
    image_with_boxes = image.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red boxes

    # Save the processed image
    cv2.imwrite(save_path, image_with_boxes)
    print(f"Saved: {save_path}")

# Example usage
base_folder = "Data/conceptual_captions_data/processed_success"  # Replace with the path to your folder
process_subfolders(base_folder)
