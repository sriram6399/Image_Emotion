import requests
import pandas as pd
import os
import json
from zipfile import ZipFile

def download_coco_captions(url, save_path):
    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Downloading the annotations file
    response = requests.get(url)
    response.raise_for_status()  # Will raise an exception for HTTP errors

    # Writing the ZIP data to a file
    zip_path = save_path + '.zip'
    with open(zip_path, 'wb') as file:
        file.write(response.content)

    # Extracting the ZIP file
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(save_path))

    print(f"Downloaded and extracted annotations to {os.path.dirname(save_path)}")

def extract_coco_captions(json_dir, csv_path):
    all_captions = []

    # Process each JSON file in the directory that starts with "captions"
    for filename in os.listdir(json_dir):
        if filename.startswith('captions') and filename.endswith('.json'):
            json_path = os.path.join(json_dir, filename)
            
            # Load JSON content
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Extract captions and image IDs
            captions = [{'image_id': item['image_id'], 'caption': item['caption']} for item in data['annotations']]
            all_captions.extend(captions)

    # Convert to DataFrame
    df = pd.DataFrame(all_captions)

    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"All captions saved to {csv_path}")

# URL to the COCO Captions annotations file
annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
annotations_path = "Data/coco_annotations"  # Adjust this path as needed
captions_csv_path = "Data/coco_captions.csv"  # Adjust this path as needed

# Download and extract captions
#download_coco_captions(annotations_url, annotations_path)
#extract_captions(annotations_path, captions_csv_path)

import pandas as pd
from datasets import load_dataset

def load_and_save_cc_dataset(dataset_name):
    # Load the dataset
    dataset = load_dataset(dataset_name, "unlabeled")
    
    # Extract 'img_id' and 'caption' from the dataset
    img_url = [item['image_url'] for item in dataset['train']]
    captions = [item['caption'] for item in dataset['train']]

    # Create a DataFrame
    df = pd.DataFrame({
        'img_url': img_url,
        'caption': captions
    })

    # Save the DataFrame to a CSV file
    csv_file_path = f'Data/conceptual_captions.csv'
    df.to_csv(csv_file_path, index=False)
    print(f'Dataset saved to {csv_file_path}')

def load_and_save_flickr_dataset(dataset_name):
    # Load the dataset
    dataset = load_dataset(dataset_name)
    
    # Extract 'img_id' and 'caption' from the dataset
    img_url = [item['img_id'] for item in dataset['test']]
    captions = [item['caption'] for item in dataset['test']]

    # Create a DataFrame
    df = pd.DataFrame({
        'image_id': img_url,
        'caption': captions
    })

    # Save the DataFrame to a CSV file
    csv_file_path = f'Data/flickr_30k_captions.csv'
    df.to_csv(csv_file_path, index=False)
    print(f'Dataset saved to {csv_file_path}')

load_and_save_cc_dataset("google-research-datasets/conceptual_captions")
load_and_save_flickr_dataset("nlphuji/flickr30k")


