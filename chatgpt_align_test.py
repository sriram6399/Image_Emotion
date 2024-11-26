import os
import json
import csv
import openai
from openai import OpenAI
from PIL import Image
import base64


openai_api_key = ""

def image_to_base64(image_path):

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

def ask_chatgpt_for_alignment(text_prompt, image_path):
 
    
    base64_image = image_to_base64(image_path)

    query = f"""
    The following task involves evaluating whether a generated image aligns with the given input text prompt.

    Prompt: "{text_prompt}"

    The image is attached

    Please analyze the image and determine whether it aligns with the given prompt. 

    Respond strictly in the following format:
    Answer: Yes/No
    Reason: [Provide a short explanation of why the image aligns or does not align with the prompt.]
    
    Donot provide any additional responses like here is the solution, the answer is as follows and so on.
    
    An example response is as follows.
    Answer: Yes
    Reason: The image shows a group of people playing soccer in a park under a sunny sky, which aligns with the given prompt.

    """

    try:
        response = OpenAI(api_key=openai_api_key).chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {
                            "url":  f"data:image/jpeg;base64,{base64_image}"
                        },
                        },
                    ],
                }
                ]
            )
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
        

    full_response = response.choices[0].message.content
    print(image_path)
    print(full_response)
    yes_or_no = "Yes" if "Answer: Yes" in full_response else "No"
    reason = full_response.split("Reason:")[1].strip() if "Reason:" in full_response else "No reason provided."
    return {"yes_or_no": yes_or_no, "reason": reason}
  
        
        

def process_subfolders(base_folder, output_csv):

    results = []

    for root, dirs, files in os.walk(base_folder):
        if "metadata.json" in files:
            # Load metadata.json
            metadata_file = os.path.join(root, "metadata.json")
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            caption = metadata.get("caption")
            caption_opp = metadata.get("caption_opp")
            image_path = os.path.join(root, metadata.get("sd_generated_image_path", ""))
            image_opp_path = os.path.join(root, metadata.get("sd_generated_image_path_opp", ""))

            if not all([os.path.exists(p) for p in [image_path, image_opp_path]]):
                print(f"Missing images in {root}. Skipping...")
                continue

            response1 = ask_chatgpt_for_alignment(caption, image_path)
            if response1 is None:
                continue
            
            response2 = ask_chatgpt_for_alignment(caption_opp, image_opp_path)
            if response2 is None:
                continue
            
            results.append({
                "text": caption,
                "image_path": image_path,
                "Image_align_yes_or_no": response1["yes_or_no"],
                "Image_align_desc": response1["reason"]
            })
            results.append({
                "text": caption_opp,
                "image_path": image_opp_path,
                "Image_align_yes_or_no": response2["yes_or_no"],
                "Image_align_desc": response2["reason"]
            })

    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["text", "image_path", "Image_align_yes_or_no", "Image_align_desc"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_csv}")


base_folder = "Data/conceptual_captions_data/processed_success"
output_csv = "Results/conceptual_alignment_results.csv"
process_subfolders(base_folder, output_csv)
