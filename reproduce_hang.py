
import sys
import time
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.append('src')

from model.semantic_draw import SemanticDraw as SemanticDrawRCCA
from util import seed_everything, get_torch_device

def reproduction():
    device = get_torch_device()
    seed = 2024
    seed_everything(seed)
    
    print(f'[INFO] Initialized with seed: {seed}')
    print(f'[INFO] Initialized with device: {device}')
    
    # Mock masks (random noise) to avoid loading files
    width, height = 512, 512
    mask_p1 = Image.new('L', (width, height), 128)
    mask_p2 = Image.new('L', (width, height), 200)
    
    mask_p1_t = T.ToTensor()(mask_p1)[-1:]
    mask_p2_t = T.ToTensor()(mask_p2)[-1:]
    background_t = torch.logical_and(mask_p1_t == 0, mask_p2_t == 0) # This might be all false effectively
    
    print("Initializing SemanticDrawRCCA...")
    streamer = SemanticDrawRCCA(
        device,
        height=height,
        width=width,
        cfg_type="none",
        autoflush=True,
        use_tiny_vae=True,
        mask_type='continuous',
        seed=seed,
        hf_key='ironjr/BlazingDriveV11m',
        has_i2t=False,
    )
    print("Initializtion Done.")
    
    print("Updating Background...")
    streamer.update_background(Image.new(size=(width, height), mode='RGB', color=(255, 255, 255)))
    print("Background Updated.")
    
    print("Updating Layer 0...")
    streamer.update_single_layer(0, 'a photo of Mount Olympus', '', background_t, 1.0, 0.0, 1.0)
    print("Layer 0 Updated.")
    
    print("Updating Layer 1...")
    streamer.update_single_layer(1, 'Greek god Zeus looking at viewer', '', mask_p1_t, 1.0, 0.0, 1.0)
    print("Layer 1 Updated.")
    
    print("Updating Layer 2...")
    streamer.update_single_layer(2, 'a small, sitting eagle', '', mask_p2_t, 1.0, 0.0, 1.0)
    print("Layer 2 Updated.")
    
    print("Warming up (Running streamer)...")
    for i in range(3):
        print(f"Streamer call {i+1}...")
        streamer()
        print(f"Streamer call {i+1} Done.")

if __name__ == "__main__":
    reproduction()
