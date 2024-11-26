import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import json
from nltk.sentiment import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
import uuid
from PIL import UnidentifiedImageError

def download_image(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            try:
                return Image.open(BytesIO(response.content))
            except UnidentifiedImageError:
                print(f"Cannot identify image file from {image_url}")
                return None
        else:
            print(f"Failed to download image: {image_url}, Status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Request failed for {image_url}: {e}")
        return None

def save_image(image, path):
    """Save an image to the specified path."""
    if image:
        image.save(path)
        print(f"Saved image to {path}")  # Progress print

def sentiment_analysis(caption):
    """Return sentiment score for the given caption using VADER."""
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(caption)['compound']
    return sentiment_score

def coco_caption_read(csv_file_path, output_dir):
    # Read the CSV file
    data = pd.read_csv(csv_file_path)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect tasks for concurrent download
    download_tasks = []
    for _, row in data.iterrows():
        image_id = row['image_id']
        caption = row['caption']
        sentiment_score = sentiment_analysis(caption)

        if abs(sentiment_score) > 0.8:
            image_url = f"http://images.cocodataset.org/train2017/{image_id:012d}.jpg"
            download_tasks.append((image_url, image_id, caption, sentiment_score))
            

    # Download images concurrently
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(lambda task: download_image(task[0]), download_tasks)

        for result, task in zip(results, download_tasks):
            image = result
            image_url,image_id, caption, sentiment_score = task[0], task[1], task[2], task[3]
            if image:
                base_dir = f"{output_dir}/{image_id}"
                os.makedirs(base_dir, exist_ok=True)
                image_path = f"{base_dir}/{image_id}_image.jpg"
                save_image(image, image_path)
                metadata = {
                    'image_id': image_id,
                    'caption': caption,
                    'sentiment_score': sentiment_score,
                    'image_path': image_path
                }
                metadata_path = f"{base_dir}/metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                print(f"Metadata saved to {metadata_path}")  # Progress print

def flickr_30k_read(output_dir):
    # Load the Flickr30k dataset from Hugging Face Hub
    dataset = load_dataset("nlphuji/flickr30k")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tasks = []
    # Iterate over each image entry in the dataset
    for data in dataset['test']:
        image_id = data['img_id']
        image_bytes = data['image']
        # Process each caption individually
        for caption in data['caption']:
            sentiment_score = sentiment_analysis(caption)
            if abs(sentiment_score) > 0.8:
                tasks.append((image_id, caption, image_bytes, sentiment_score))
                

    # Process tasks
    for task in tasks:
        image_id, caption, image_bytes, sentiment_score = task
        base_dir = f"{output_dir}/{image_id}"
        os.makedirs(base_dir, exist_ok=True)

        # Save the image only if it does not already exist to avoid redundancy
        image_path = f"{base_dir}/{image_id}_image.jpg"
        if not os.path.exists(image_path):
            save_image(image_bytes, image_path)

        # Create metadata for each caption
        metadata_path = f"{base_dir}/metadata.json"  # Use hash to uniquely identify captions
        metadata = {
            'image_id': image_id,
            'caption': caption,
            'sentiment_score': sentiment_score,
            'image_path': image_path
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to {metadata_path}")  # Progress print
        
def generate_unique_id(output_dir):
    """Generate a unique identifier that does not already exist in the output directory."""
    while True:
        unique_id = str(uuid.uuid4())
        if not os.path.exists(os.path.join(output_dir, unique_id)):
            return unique_id
        
def conceptual_captions_read(output_dir):
    """Process and save images and metadata for the Conceptual Captions dataset."""
    dataset = load_dataset("google-research-datasets/conceptual_captions", "unlabeled")

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for data in dataset['train']:
        try:
            caption = data['caption']
            sentiment_score = sentiment_analysis(caption)
            
            if abs(sentiment_score) > 0.85:
                print("Processing   :",caption)
                image_url = data['image_url']
                unique_id = generate_unique_id(output_dir) 
    
                image = download_image(image_url)
                
                print("Processing Id:",unique_id)
                if image:
                    base_dir = f"{output_dir}/{unique_id}"
                    os.makedirs(base_dir, exist_ok=True)
                    image_path = f"{base_dir}/{unique_id}_image.jpg"
                    save_image(image, image_path)
                    metadata = {
                        'image_id': unique_id,
                        'caption': caption,
                        'sentiment_score': sentiment_score,
                        'image_path': image_path
                    }
                    metadata_path = f"{base_dir}/metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
                    print(f"Metadata saved to {metadata_path}")
                    
        except Exception as e:
            print(e)
            continue

       
        
# Example usage for Flickr30K
output_dir = "Data/flickr30k_images_data"
flickr_30k_read(output_dir)

# Example usage for COCO
csv_file_path = "Data/coco_captions.csv"
output_dir = "Data/coco_images_data"
coco_caption_read(csv_file_path, output_dir)

# Example usage
output_dir = "Data/conceptual_captions_data"
conceptual_captions_read(output_dir)