import os
import cv2
import json
import csv
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load a pre-trained ResNet model
model = models.resnet50(pretrained=True)
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

def extract_features(image):
    input_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        features = model(input_tensor)
    return features

def compute_similarity(image1, image2):
    features1 = extract_features(image1)
    features2 = extract_features(image2)
    similarity = F.cosine_similarity(features1, features2).item()
    return similarity



def extract_contours(heatmap_path, image_path):
    heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)

    # Compute percentile-based threshold (e.g., 95th percentile)
    threshold_value = np.percentile(heatmap, 95)  # Top 5% brightest pixels
    

    # Apply binary thresholding
    _, binary_heatmap = cv2.threshold(heatmap, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contour_image = image[y:y+h, x:x+w]
        contour_images.append(contour_image)
    return contour_images

    
    
    

def process_subfolders(base_folder, output_csv, threshold, similarity_range):
    results = []

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

     

            # Extract contours from both images
            contour_images1 = extract_contours(heatmap_path, image_path)
            contour_images2 = extract_contours(heatmap_opp_path, image_opp_path)

            similarities = []
            for img1 in contour_images1:
                highest_similarity = 0
                for img2 in contour_images2:
                    # Convert OpenCV image to PIL for ResNet compatibility
                    img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
                    img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

                    # Compute similarity
                    similarity = compute_similarity(img1_pil, img2_pil)
                    highest_similarity = max(highest_similarity, similarity)

                # Append the highest similarity if it exceeds the threshold
                if highest_similarity >= threshold:
                    similarities.append(highest_similarity)

            # Compute the average similarity for valid matches
            average_similarity = sum(similarities) / len(similarities) if similarities else 0

            # Determine the label
            label = "yes" if similarity_range[0] <= average_similarity <= similarity_range[1] else "no"

            # Append results
            results.append({
                "contour_image_path": os.path.join(root, f"{image_id}_contour_boxes.png"),
                "contour_image_opp_path": os.path.join(root, f"{image_id}_contour_boxes_opp.png"),
                "average_similarity": average_similarity,
                "label": label
            })

    # Write results to CSV
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["contour_image_path", "contour_image_opp_path", "average_similarity", "label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_csv}")

# Example usage
base_folder = "Data/coco_images_data/processed_success"  # Replace with your folder path
output_csv = "Results/coco_contour_similarity_results.csv"
process_subfolders(base_folder, output_csv, threshold=0.6, similarity_range=(0.7, 0.9))
