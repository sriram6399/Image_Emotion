import os
import json
import csv
from transformers import pipeline
import torch
import clip
from PIL import Image

# ArtEmis emotion labels
emotion_labels = ["amusement", "awe", "contentment", "excitement", "anger", "fear", "sadness", "surprise"]

# Load the GoEmotions text classifier
text_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Load CLIP for image classification
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Map GoEmotions labels to ArtEmis labels
goemotions_to_artemis = {
    "joy": "amusement",
    "admiration": "awe",
    "love": "contentment",
    "excitement": "excitement",
    "anger": "anger",
    "fear": "fear",
    "sadness": "sadness",
    "surprise": "surprise"
}

# Function to classify text emotion using GoEmotions and map to ArtEmis labels
def classify_text_emotion(prompt):
    # Run text classification
    results = text_classifier(prompt, return_all_scores=True)
    
    # Debugging: Print results to inspect structure
    print(f"Results for prompt '{prompt}': {results}")
    
    # Ensure results is a list of dictionaries
    if isinstance(results, list) and isinstance(results[0], list):  # If wrapped in an extra list
        results = results[0]
    
    # Sort results by score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Map the highest-scoring label to ArtEmis labels
    goemotion_label = results[0]["label"].lower()
    artemis_emotion = goemotions_to_artemis.get(goemotion_label, "unknown")
    confidence = results[0]["score"]
    return artemis_emotion, confidence

# Function to classify image emotion using CLIP
def classify_image_emotion(image_path):
    # Preprocess and encode image
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(clip.tokenize(emotion_labels).to(device))
    
    # Compute similarity scores
    logits_per_image = image_features @ text_features.T
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
    # Find the most likely emotion
    top_idx = probs.argmax()
    return emotion_labels[top_idx], probs[0][top_idx]

# Function to process paths and generate results
def process_paths_and_store_results(paths, output_csv):
    results = []

    for base_path in paths:
        for root, dirs, files in os.walk(base_path):
            print(f"Processing folder: {root}")
            # Check if metadata.json exists in the current directory
            if "metadata.json" in files:
                metadata_file = os.path.join(root, "metadata.json")
                
                # Load metadata.json
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                # Extract paths and captions
                caption = metadata.get("caption")
                caption_opp = metadata.get("caption_opp")
                image_path = os.path.join(root, metadata.get("sd_generated_image_path", ""))
                image_opp_path = os.path.join(root, metadata.get("sd_generated_image_path_opp", ""))

                # Debugging: Print metadata content
                print(f"Metadata content: {metadata}")

                # Classify emotions for caption and image
                text_emotion, text_confidence = classify_text_emotion(caption)
                image_emotion, image_confidence = classify_image_emotion(image_path)
                classifier_equivalence = "yes" if text_emotion == image_emotion else "no"

                # Classify emotions for caption_opp and image_opp
                text_emotion_opp, text_confidence_opp = classify_text_emotion(caption_opp)
                image_emotion_opp, image_confidence_opp = classify_image_emotion(image_opp_path)
                classifier_equivalence_opp = "yes" if text_emotion_opp == image_emotion_opp else "no"

                # Append results
                results.append({
                    "caption": caption,
                    "caption_opp": caption_opp,
                    "image_path": image_path,
                    "image_opp_path": image_opp_path,
                    "classifier_equivalence": classifier_equivalence,
                    "classifier_equivalence_opp": classifier_equivalence_opp,
                })

    # Write results to a CSV file
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["caption", "caption_opp", "image_path", "image_opp_path", "classifier_equivalence", "classifier_equivalence_opp"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

# Example usage
paths = [
    "Data/conceptual_captions_data/processed_success"
]  # List of base paths to process
output_csv = "Results/conceptual_classification_results.csv"

process_paths_and_store_results(paths, output_csv)
