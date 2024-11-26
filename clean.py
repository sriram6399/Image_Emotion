import os
import json

def clean_metadata_if_missing_images(base_folder):
    # List of keys to delete if the image path does not exist
    keys_to_delete = [
        'significant_word', 
        'significant_word_opp', 
        'caption_opp',
        'sd_generated_image_path', 
        'sd_attention_overlay_path', 
        'sd_heatmap', 
        'sd_generated_image_path_opp', 
        'sd_attention_overlay_path_opp', 
        'sd_heatmap_opp'
    ]
    
    # Iterate over each subfolder in the base folder
    for subfolder in [os.path.join(base_folder, f) for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]:
        metadata_path = os.path.join(subfolder, "metadata.json")
        
        # Check if metadata.json exists in the subfolder
        if not os.path.isfile(metadata_path):
            print(f"No metadata.json found in {subfolder}. Skipping...")
            continue

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check if sd_attention_overlay_path is in metadata and the file exists
        sd_attention_overlay_path = metadata.get('sd_attention_overlay_path')
        if sd_attention_overlay_path:
            image_path = os.path.join(subfolder, sd_attention_overlay_path)
            if not os.path.isfile(image_path):
                print(f"Image missing at {image_path}. Cleaning metadata in {subfolder}.")
                
                # Delete specified keys from metadata
                for key in keys_to_delete:
                    if key in metadata:
                        del metadata[key]
                
                # Save the updated metadata back to metadata.json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
            else:
                print(f"Image found at {image_path}. No changes made in {subfolder}.")
        else:
            print(f"'sd_attention_overlay_path' not present in metadata for {subfolder}. No changes made.")

# Usage
base_folder = "Data/conceptual_captions_input"
#clean_metadata_if_missing_images(base_folder)

import os
import shutil
import json

def traverse_and_process(base_folder, new_subfolder_name):
    # Define patterns and their replacements
    replacement_patterns = {
    "_sd_imageopposite.png": "_sd_image_opp.png",
    "_sd_overlayopposite.png": "_sd_overlay_opp.png",
    "_sd_heatmapopposite.png": "_sd_heatmap_opp.png",
    "_sd_imageoriginal.png": "_sd_image.png",
    "_sd_overlayoriginal.png": "_sd_overlay.png",
    "_sd_heatmaporiginal.png": "_sd_heatmap.png",
    }

    # Create the destination folder if it doesn't exist
    destination_folder = os.path.join(base_folder, new_subfolder_name)
    os.makedirs(destination_folder, exist_ok=True)

    for root, dirs, files in os.walk(base_folder):
        # Skip the destination folder itself during traversal
        if root.startswith(destination_folder):
            continue

        # Replace duplicates with original suffix
        for duplicate_suffix, original_suffix in replacement_patterns.items():
            for file in files:
                # If a duplicate is found
                if file.endswith(duplicate_suffix):
                    # Generate the new filename with the original suffix
                    new_filename = file.replace(duplicate_suffix, original_suffix)
                    duplicate_path = os.path.join(root, file)
                    new_path = os.path.join(root, new_filename)

                    # Delete any existing file with the original suffix
                    if os.path.exists(new_path):
                        os.remove(new_path)
                        print(f"Deleted existing file with original suffix: {new_path}")

                    # Rename the duplicate file to the original suffix
                    os.rename(duplicate_path, new_path)
                    print(f"Renamed '{duplicate_path}' to '{new_path}'")

        # Check if metadata.json exists and contains the key "caption_opp"
        metadata_file_path = os.path.join(root, "metadata.json")
        if os.path.isfile(metadata_file_path):
            with open(metadata_file_path, 'r') as metadata_file:
                try:
                    metadata = json.load(metadata_file)
                    if "caption_opp" in metadata:
                        # Move the entire subfolder to the new destination folder
                        shutil.move(root, os.path.join(destination_folder, os.path.basename(root)))
                        print(f"Moved folder '{root}' to '{destination_folder}'")
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON in file: {metadata_file_path}")
                except Exception as e:
                    print(f"Error processing file {metadata_file_path}: {e}")

# Base folder to start traversal
base_folder = "Data/conceptual_captions_data"
# Name of the new subfolder where matching folders will be moved
new_subfolder_name = "processed_success"

traverse_and_process(base_folder, new_subfolder_name)

import os
import csv

# Define the mapping of old filenames to new filenames
file_rename_map = {
    "_sd_imageopposite.png": "_sd_image_opp.png",
    "_sd_overlayopposite.png": "_sd_overlay_opp.png",
    "_sd_heatmapopposite.png": "_sd_heatmap_opp.png",
    "_sd_imageoriginal.png": "_sd_image.png",
    "_sd_overlayoriginal.png": "_sd_overlay.png",
    "_sd_heatmaporiginal.png": "_sd_heatmap.png",
}

def rename_files_with_mapping(base_folder):
    changes = []  # To store old and new file paths

    # Walk through all subfolders and files in the directory
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            # Check if the file matches any key in the mapping
            for old_suffix, new_suffix in file_rename_map.items():
                if file.endswith(old_suffix):
                    # Construct the full file paths
                    old_path = os.path.join(root, file)
                    new_file = file.replace(old_suffix, new_suffix)
                    new_path = os.path.join(root, new_file)

                    # Rename the file
                    os.rename(old_path, new_path)

                    # Store the change in the list
                    changes.append({"old_path": old_path, "new_path": new_path})
                    print(f"Renamed: {old_path} -> {new_path}")

    

# Example usage
base_folder = "Data/coco_images_data/processed_success"  # Replace with the path to your folder

#rename_files_with_mapping(base_folder)




