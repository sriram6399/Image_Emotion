import os
import json
import sys
import torch
from diffusers import DiffusionPipeline
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import yaml
import numpy as np
import matplotlib.pyplot as plt
from daam import trace
# Add the taming-transformers path
sys.path.append('taming-transformers')
from taming.models.vqgan import VQModel

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Set multiprocessing start method to spawn for CUDA compatibility
torch.multiprocessing.set_start_method("spawn", force=True)


'''model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
device = 'cuda'

pipe = DiffusionPipeline.from_pretrained(model_id, use_auth_token=True, torch_dtype=torch.float16, use_safetensors=True, variant='fp16')
pipe = pipe.to(device)

prompt = 'The man is holding a naked body which is sad'
gen = set_seed(0)  # for reproducibility

with torch.no_grad():
    with trace(pipe) as tc:
        out = pipe(prompt, num_inference_steps=50, generator=gen)
        heat_map = tc.compute_global_heat_map()
        heat_map = heat_map.compute_word_heat_map('sad')
        heat_map.plot_overlay(out.images[0])
        plt.savefig("Results/test.png")
        plt.show()'''
from accelerate import dispatch_model
from diffusers import DiffusionPipeline
import torch

# Load the pre-trained DiffusionPipeline normally
sd_pipe = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant='fp16'
)



def distribute_pipeline_components(sd_pipe, gpu_ids):
    devices = [torch.device(f"cuda:{gpu_id}") for gpu_id in gpu_ids]
    num_devices = len(devices)

    # Distribute VAE submodules across GPUs
    print("\nDistributing VAE modules:")
    vae_modules = list(sd_pipe.vae.encoder.children()) + list(sd_pipe.vae.decoder.children())
    for i, module in enumerate(vae_modules):
        target_device = devices[1]
        print(f"  Moving VAE module {i} to {target_device}")
        module.to(target_device)

    # Distribute UNet submodules across GPUs
    print("\nDistributing UNet modules:")
    unet_modules = list(sd_pipe.unet.children())
    for i, block in enumerate(unet_modules):
        target_device = devices[0]
        print(f"  Moving UNet block {i} to {target_device}")
        block.to(target_device)

    # Assign other components that can be moved to GPU
    print("\nAssigning additional components to GPUs:")

    # Text Encoder
    target_device = devices[1]
    print(f"  Moving text_encoder to {target_device}")
    sd_pipe.text_encoder.to(target_device)

    # Text Encoder 2
    target_device = devices[1]
    print(f"  Moving text_encoder_2 to {target_device}")
    sd_pipe.text_encoder_2.to(target_device)

    # Image Encoder
    # target_device = devices[2 % num_devices]
    # print(f"  Moving image_encoder to {target_device}")
    # sd_pipe.image_encoder.to(target_device)

    # Feature Extractor
    # target_device = devices[3 % num_devices]
    # print(f"  Moving feature_extractor to {target_device}")
    # sd_pipe.feature_extractor.to(target_device)

    # Note: Tokenizers and Scheduler typically remain on the CPU as they do not require GPU acceleration.
    print("\nKeeping scheduler, tokenizer, and tokenizer_2 on CPU as they don't require GPU.")

    print("\nDistribution complete.")
    return sd_pipe

sd_pipe = distribute_pipeline_components(sd_pipe,[1,2,3])
        
def generate_image_with_stable_diffusion(pipe, prompt, word):
    
    with trace(pipe) as tc:
        output = pipe(prompt,num_inference_steps=50)
  
    
    heat_map = tc.compute_global_heat_map()
    heat_map = heat_map.compute_word_heat_map(word)
    heat_map.plot_overlay(output.images[0])
    plt.savefig("Results/test.png")
    plt.show()
    return output.images[0], tc.compute_global_heat_map().compute_word_heat_map(word)

generate_image_with_stable_diffusion(sd_pipe,"happy woman enjoying nature beautiful blonde walking on the field .", "enjoying")

'''from torchvision.transforms.functional import normalize, to_pil_image
from torchvision import transforms

def load_vqgan_model(config_path, checkpoint_path, device):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    ddconfig, lossconfig = config["model"]["params"]["ddconfig"], config["model"]["params"]["lossconfig"]
    n_embed, embed_dim = config["model"]["params"]["n_embed"], config["model"]["params"]["embed_dim"]
    vqgan_model = VQModel(ddconfig, lossconfig, n_embed, embed_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vqgan_model.load_state_dict(checkpoint["state_dict"])
    return vqgan_model



vqgan_model = load_vqgan_model(
        "taming-transformers/vqgan_imagenet_f16_16384/configs/model.yaml", 
        "taming-transformers/vqgan_imagenet_f16_16384/ckpts/last.ckpt", 
        f"cuda:1"
    )

def generate_image_with_vqgan(vqgan_model, clip_model, clip_processor, prompt, word, device):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 pixels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    text_inputs = clip_processor(text=[prompt, word], return_tensors="pt", padding=True).to(device)
    latent = torch.randn(1, 256, 16, 16, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([latent], lr=0.1)

    for i in range(100):
        optimizer.zero_grad()
        generated = vqgan_model.decode(latent).squeeze(0)
        generated_image = to_pil_image(generated.cpu())
        transformed_image = preprocess(generated_image).unsqueeze(0).to(device)

        image_features = clip_model.get_image_features(pixel_values=transformed_image)
        text_features = clip_model.get_text_features(**text_inputs)

        loss = 1 - torch.cosine_similarity(text_features[0], image_features, dim=-1)
        loss.mean().backward()
        optimizer.step()

    image_features = image_features.detach()
    word_features = text_features[1].detach()

    cos_sim = torch.cosine_similarity(image_features, word_features, dim=1)
    sim_map = cos_sim.view(1, -1).cpu().numpy()  # Ensuring correct view
    sim_map = np.kron(sim_map, np.ones((224, 224)))  # Scale up to the image size

    final_image = to_pil_image(transformed_image.squeeze(0).cpu())
    image_np = np.array(final_image)

    # Visualizing both images with different scales to differentiate
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title('Generated Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_np, alpha=0.5)
    plt.imshow(sim_map, cmap='hot', alpha=0.5)
    plt.title(f'Heatmap for "{word}"')
    plt.axis('off')
    plt.savefig("Results/Generated_Image_and_Heatmap_Adjusted.png")
    plt.show()

    return final_image, sim_map


clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(f"cuda:1")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
prompt = "A serene beach at sunset"
word = "sunset"
generate_image_with_vqgan(vqgan_model, clip_model, clip_processor,prompt,word,"cuda:1")'''