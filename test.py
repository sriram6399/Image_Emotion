import torch
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

'''def find_significant_word_with_shap(sentence, model_name="SamLowe/roberta-base-go_emotions"):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Ensure the model has `label2id` and `id2label` attributes as integers
    model.config.label2id = {str(i): i for i in range(model.config.num_labels)}
    model.config.id2label = {i: str(i) for i in range(model.config.num_labels)}

    # Create a transformers pipeline for text classification
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

    # Predict the label for the sentence to get the predicted label index
    prediction = pipe(sentence)
    
    predicted_label_index = int(torch.argmax(torch.tensor([p["score"] for p in prediction[0]])))
    print(prediction[0][predicted_label_index])
    # Define the SHAP explainer with the pipeline
    explainer = shap.Explainer(pipe)

    # Get SHAP values for the sentence
    shap_values = explainer([sentence])

    # Tokenize the sentence and extract SHAP values for each token
    tokens = tokenizer.tokenize(sentence)
    word_shap_values = shap_values[0].values  # SHAP values for each token, across classes

    # Use only the SHAP values for the predicted label index
    word_shap_scores = [values[predicted_label_index] for values in word_shap_values]

    # Pair each token with its SHAP value for the predicted label and find the most significant word
    token_shap_pairs = list(zip(tokens, word_shap_scores))
    token_shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    most_significant_word = token_shap_pairs[0][0]
    
    return most_significant_word'''

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def find_significant_word_with_masking(sentence, model_name="monologg/bert-base-cased-goemotions-original"):
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Get the original prediction probabilities
    with torch.no_grad():
        original_output = model(**inputs)
        original_probs = torch.nn.functional.softmax(original_output.logits, dim=-1)[0]

    # Store the initial predicted label and its probability
    predicted_label = torch.argmax(original_probs).item()
    original_prob = original_probs[predicted_label].item()

    # Iterate through each token, mask it, and get the impact on prediction
    max_drop = 0
    most_significant_word = None

    for i, token in enumerate(tokens):
        # Skip special tokens ([CLS] and [SEP])
        if token in ["[CLS]", "[SEP]"]:
            continue

        # Create a new input with the current token replaced by [MASK]
        masked_input_ids = inputs["input_ids"].clone()
        masked_input_ids[0, i] = tokenizer.mask_token_id
        masked_inputs = {"input_ids": masked_input_ids, "attention_mask": inputs["attention_mask"]}

        # Get the prediction probabilities for the masked input
        with torch.no_grad():
            masked_output = model(**masked_inputs)
            masked_probs = torch.nn.functional.softmax(masked_output.logits, dim=-1)[0]

        # Check if the predicted label changes
        new_label = torch.argmax(masked_probs).item()
        if new_label != predicted_label:
            # Calculate the probability drop for the original predicted label
            masked_prob = masked_probs[predicted_label].item()
            drop = original_prob - masked_prob

            # Update if this word causes the highest probability drop with a label change
            if drop > max_drop:
                max_drop = drop
                most_significant_word = token

    return most_significant_word

# Example usage


import torch
from diffusers import DiffusionPipeline
from transformers import  AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
from transformers import set_seed
import numpy as np
import matplotlib.pyplot as plt
from daam import trace




model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
device = 'cuda'

pipe = DiffusionPipeline.from_pretrained(model_id, use_auth_token=True, torch_dtype=torch.float16, use_safetensors=True, variant='fp16')
pipe = pipe.to(device)
 
# Example usage
sentence = """An exhilarating moment of freedom and love as Rose spreads her arms like wings on the ship's bow, her spirit soaring in Jack's steady embrace from behind, against the endless horizon of the open sea."""
significant_word = find_significant_word_with_masking(sentence)
print("Most significant word:", significant_word)
gen = set_seed(0)  # for reproducibility

with torch.no_grad():
    with trace(pipe) as tc:
        out = pipe(sentence, num_inference_steps=50, generator=gen)
        out.images[0].save("Results/test_img.png")
        heat_map = tc.compute_global_heat_map()
        heat_map = heat_map.compute_word_heat_map(significant_word)
        heat_map.plot_overlay(out.images[0])
        plt.axis('off')
        plt.savefig("Results/test.png")
        plt.show()
        
        
sentence2 = """An exhilarating moment of freedom and hate as Rose spreads her arms like wings on the ship's bow, her spirit soaring in Jack's steady embrace from behind, against the endless horizon of the open sea."""
significant_word2 = "hate"
gen = set_seed(0)  # for reproducibility

with torch.no_grad():
    with trace(pipe) as tc:
        out = pipe(sentence2, num_inference_steps=50, generator=gen)
        out.images[0].save("Results/test_img2.png")
        heat_map = tc.compute_global_heat_map()
        heat_map = heat_map.compute_word_heat_map(significant_word2)
        heat_map.plot_overlay(out.images[0])
        plt.axis('off')
        plt.savefig("Results/test2.png")
        
        plt.show()
        
        
sentence3 = """A Dog is running happily in the field"""
significant_word3 = find_significant_word_with_masking(sentence3)
print("Most significant word:", significant_word3)
gen = set_seed(0)  # for reproducibility

with torch.no_grad():
    with trace(pipe) as tc:
        out = pipe(sentence3, num_inference_steps=50, generator=gen)
        out.images[0].save("Results/test_img3.png")
        heat_map = tc.compute_global_heat_map()
        heat_map = heat_map.compute_word_heat_map(significant_word3)
        heat_map.plot_overlay(out.images[0])
        plt.axis('off')
        plt.savefig("Results/test3.png")
        plt.show()
        
        
sentence4 = """A Dog is running sadly in the field"""
significant_word4 = "sadly"
gen = set_seed(0)  # for reproducibility

with torch.no_grad():
    with trace(pipe) as tc:
        out = pipe(sentence4, num_inference_steps=50, generator=gen)
        out.images[0].save("Results/test_img4.png")
        heat_map = tc.compute_global_heat_map()
        heat_map = heat_map.compute_word_heat_map(significant_word4)
        heat_map.plot_overlay(out.images[0])
        plt.axis('off')
        plt.savefig("Results/test4.png")
        
        plt.show()