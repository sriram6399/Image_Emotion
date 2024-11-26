import os
import json
import sys
from matplotlib import pyplot as plt
import torch
from diffusers import DiffusionPipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
from daam import trace
import numpy as np
import nltk
from nltk.corpus import wordnet
from typing import Tuple, Optional

# Environment setup
nltk.download('wordnet')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.multiprocessing.set_start_method("spawn", force=True)

# Helper function to find antonyms using WordNet
def get_antonym(word: str) -> Optional[str]:
    antonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
    return antonyms[0] if antonyms else None

# Function to replace a word in a sentence with its antonym
def replace_with_opposite(sentence: str, word: str) -> Tuple[Optional[str], str]:
    opposite_word = get_antonym(word)
    if opposite_word is None:
        return None, "Antonym not found in WordNet."
    modified_sentence = sentence.replace(word, opposite_word, 1)
    return opposite_word, modified_sentence

# Load models and processors
def load_models_and_processors(gpu_id):
    try:
        sd_pipe = DiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-xl-base-1.0',
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant='fp16',
            device_map='balanced'
        )
        bert_tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
        bert_model = AutoModelForSequenceClassification.from_pretrained("monologg/bert-base-cased-goemotions-original").to(f"cuda:{gpu_id[1]}")
    except Exception as e:
        print(f"Failed to load models: {e}")
        return None
    return sd_pipe, bert_tokenizer, bert_model

# Find significant word by masking
def find_significant_word_with_masking(sentence, tokenizer, model, device):
    try:
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        with torch.no_grad():
            original_output = model(**inputs)
            original_probs = torch.nn.functional.softmax(original_output.logits, dim=-1)[0]
            predicted_label = torch.argmax(original_probs).item()
            original_prob = original_probs[predicted_label].item()

        max_drop, most_significant_word = 0, None
        for i, token in enumerate(tokens):
            if token in ["[CLS]", "[SEP]"]:
                continue
            masked_input_ids = inputs["input_ids"].clone()
            masked_input_ids[0, i] = tokenizer.mask_token_id
            masked_inputs = {"input_ids": masked_input_ids, "attention_mask": inputs["attention_mask"]}
            with torch.no_grad():
                masked_output = model(**masked_inputs)
                masked_probs = torch.nn.functional.softmax(masked_output.logits, dim=-1)[0]
                drop = original_prob - masked_probs[predicted_label].item()
                if drop > max_drop:
                    max_drop, most_significant_word = drop, token.replace("Ġ", "").replace("##", "")
        
        # Explicitly delete inputs tensor to release GPU memory
        del inputs
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error finding significant word: {e}")
        return None
    return most_significant_word

# Generate and save images
# Generate and save images
def generate_image(pipe, prompt, word, subfolder, image_id, suffix):
    try:
        with torch.no_grad():
            with trace(pipe) as tc:
                output = pipe(prompt, num_inference_steps=50)

        # Ensure output is generated successfully
        if not output or not output.images:
            print(f"Failed to generate output for prompt '{prompt}'.")
            del output, global_heat_map, word_heat_map
            torch.cuda.empty_cache()
            return None

        # Save paths
        image_path = os.path.join(subfolder, f"{image_id}_sd_image{suffix}.png")
        overlay_path = os.path.join(subfolder, f"{image_id}_sd_overlay{suffix}.png")
        attention_map_path = os.path.join(subfolder, f"{image_id}_sd_heatmap{suffix}.png")

        # Extract and save heat maps
        global_heat_map = tc.compute_global_heat_map()
        word_heat_map = global_heat_map.compute_word_heat_map(word)
        
        # Save the overlay and original image
        word_heat_map.plot_overlay(output.images[0], overlay_path)
        save_image(output.images[0], image_path)

        # Save heatmap as an image
        heatmap_array = (word_heat_map.expand_as(output.images[0]).clamp_(0, 1) * 255).cpu().numpy().astype(np.uint8)
        save_image(Image.fromarray(heatmap_array), attention_map_path)

        # After returning the output, clear any unused tensors
        generated_image = output.images[0]  # Store the image for returning

    except Exception as e:
        print(f"Error generating or saving images for prompt '{prompt}': {e}")
        del output, global_heat_map, word_heat_map
        torch.cuda.empty_cache()
        return None

    # Clear up unused memory
    del output, global_heat_map, word_heat_map
    torch.cuda.empty_cache()

    # Return the generated image
    return generated_image


# Save an image
def save_image(image, path):
    try:
        image.save(path)
    except Exception as e:
        print(f"Failed to save image at {path}: {e}")

# Process folder for each dataset
def process_folder(subfolder, sd_pipe, bert_tokenizer, bert_model, gpu_id):
    try:
        json_file = next((f for f in os.listdir(subfolder) if f.endswith('.json')), None)
        if not json_file:
            print(f"No JSON file found in {subfolder}. Skipping...")
            return

        metadata_path = os.path.join(subfolder, json_file)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        if 'caption_opp' in metadata:
            print(f"'caption_opp' found in metadata for {subfolder}. Skipping this folder.")
            return
        
        prompt = metadata.get('caption')
        image_id = metadata.get('image_id')
        
        if not (prompt and image_id):
            print(f"Skipping subfolder {subfolder} as 'caption' or 'image_id' is missing in metadata.")
            return

        # Find the most significant word
        significant_word = find_significant_word_with_masking(prompt, bert_tokenizer, bert_model, f"cuda:{gpu_id[1]}")
        if significant_word is None:
            print(f"No significant word found for prompt '{prompt}'. Skipping...")
            return
        
        # Replace the significant word with its antonym
        significant_word_opp, caption_opp = replace_with_opposite(prompt, significant_word)
        if significant_word_opp is None:
            print(f"No antonym found for significant word in prompt '{prompt}'. Skipping...")
            return

        # Generate original and antonym images
        gen_i = generate_image(sd_pipe, prompt, significant_word, subfolder, image_id, "")
        gen_i_opp = generate_image(sd_pipe, caption_opp, significant_word_opp, subfolder, image_id, "")
        
        if (gen_i is None) or (gen_i_opp is None):
            print("Image not generated")
            return

        # Update metadata
        metadata.update({
            'significant_word': significant_word,
            'significant_word_opp': significant_word_opp,
            'caption_opp': caption_opp,
            'sd_generated_image_path': f"{image_id}_sd_image.png",
            'sd_attention_overlay_path': f"{image_id}_sd_overlay.png",
            'sd_heatmap': f"{image_id}_sd_heatmap.png",
            'sd_generated_image_path_opp': f"{image_id}_sd_image_opp.png",
            'sd_attention_overlay_path_opp': f"{image_id}_sd_overlay_opp.png",
            'sd_heatmap_opp': f"{image_id}_sd_heatmap_opp.png",
        })

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        # Explicitly delete any accumulated tensors and clear cache
        del prompt, image_id, significant_word, significant_word_opp, caption_opp
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error processing folder {subfolder}: {e}")

# Process all subfolders
def process_subfolders(base_folder, gpu_id):
    sd_pipe, bert_tokenizer, bert_model = load_models_and_processors(gpu_id)
    if not all([sd_pipe, bert_tokenizer, bert_model]):
        print("Failed to load one or more models. Exiting...")
        return

    subfolders = [os.path.join(base_folder, subfolder) for subfolder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, subfolder))]
    for subfolder in subfolders:
        print(f"Processing {subfolder} in {base_folder}")
        process_folder(subfolder, sd_pipe, bert_tokenizer, bert_model, gpu_id)

# Run the processing for multiple datasets
if __name__ == "__main__":
    gpu_id = [1, 2, 3]
    for dataset_folder in ["Data/conceptual_captions_data"]:
        process_subfolders(dataset_folder, gpu_id)
