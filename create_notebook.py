import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# RCCA vs Baseline Comparison\n",
    "\n",
    "This notebook compares the original **Mask-Centering Bootstrapping** algorithm with the new **Region-Constrained Cross-Attention Isolation (RCCA)**.\n",
    "\n",
    "We compare:\n",
    "1. **Inference Speed (FPS)**: Should be similar or better with RCCA.\n",
    "2. **Visual Quality**: RCCA should reduce \"Center Bias\" and background artifacts."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import time\n",
    "import warnings\n",
    "import gc\n",
    "\n",
    "sys.path.append('../src')\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "import torch\n",
    "import torchvision.transforms as T\n",
    "from PIL import Image\n",
    "from diffusers.utils import make_image_grid\n",
    "from IPython.display import clear_output, display\n",
    "\n",
    "from util import seed_everything\n",
    "# Import both versions\n",
    "from model.semantic_draw_original import SemanticDraw as SemanticDrawOriginal\n",
    "from model.semantic_draw import SemanticDraw as SemanticDrawRCCA\n",
    "\n",
    "device_id = 0\n",
    "device = f'cuda:{device_id}'\n",
    "seed = 2024\n",
    "seed_everything(seed)\n",
    "print(f'[INFO] Initialized with seed  : {seed}')\n",
    "print(f'[INFO] Initialized with device: {device}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Masks (same as demo)\n",
    "print('[INFO] Loading masks...') \n",
    "mask_p1 = Image.open('../assets/zeus/prompt_p1.png').convert('RGBA')\n",
    "mask_p2 = Image.open('../assets/zeus/prompt_p2.png').convert('RGBA')\n",
    "\n",
    "mask_p1_t = T.ToTensor()(mask_p1)[-1:]\n",
    "mask_p2_t = T.ToTensor()(mask_p2)[-1:]\n",
    "background_t = torch.logical_and(mask_p1_t == 0, mask_p2_t == 0)\n",
    "\n",
    "height, width = mask_p1_t.shape[-2:]\n",
    "display(make_image_grid([mask_p1, mask_p2], 1, 2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def run_benchmark(ModelClass, title):\n",
    "    print(f\"\\n--- Running {title} ---\")\n",
    "    seed_everything(seed)\n",
    "    \n",
    "    # Initialize Model\n",
    "    streamer = ModelClass(\n",
    "        device,\n",
    "        height=height,\n",
    "        width=width,\n",
    "        cfg_type=\"none\",\n",
    "        autoflush=True,\n",
    "        use_tiny_vae=False,\n",
    "        mask_type='continuous',\n",
    "        bootstrap_steps=2, # Relevant for Original, Ignored/Deprecated for RCCA\n",
    "        seed=seed,\n",
    "    )\n",
    "    \n",
    "    # Register Background & Layers\n",
    "    streamer.update_background(Image.new(size=(width, height), mode='RGB', color=(255, 255, 255)))\n",
    "    \n",
    "    streamer.update_single_layer(0, 'a photo of Mount Olympus', '', background_t, 1.0, 0.0, 1.0)\n",
    "    streamer.update_single_layer(1, 'Greek god Zeus looking at viewer', '', mask_p1_t, 1.0, 0.0, 1.0)\n",
    "    streamer.update_single_layer(2, 'a small, sitting eagle', '', mask_p2_t, 1.0, 0.0, 1.0)\n",
    "    \n",
    "    # Warmup\n",
    "    print(\"Warming up...\")\n",
    "    for _ in range(3):\n",
    "        streamer()\n",
    "        \n",
    "    # Benchmark Loop\n",
    "    print(\"Benchmarking...\")\n",
    "    t0 = time.time()\n",
    "    n_frames = 30\n",
    "    for _ in range(n_frames):\n",
    "        img = streamer()\n",
    "    t1 = time.time()\n",
    "    \n",
    "    fps = n_frames / (t1 - t0)\n",
    "    print(f\"{title} Results:\")\n",
    "    print(f\"  FPS: {fps:.2f}\")\n",
    "    \n",
    "    # cleanup to save VRAM\n",
    "    del streamer\n",
    "    torch.cuda.empty_cache()\n",
    "    gc.collect()\n",
    "    \n",
    "    return img, fps"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run Original\n",
    "img_orig, fps_orig = run_benchmark(SemanticDrawOriginal, \"Original (Mask Centering)\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run RCCA\n",
    "img_rcca, fps_rcca = run_benchmark(SemanticDrawRCCA, \"RCCA (Attention Masking)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Comparison Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f\"Original FPS: {fps_orig:.2f}\")\n",
    "print(f\"RCCA FPS    : {fps_rcca:.2f}\")\n",
    "print(f\"Speedup     : {fps_rcca/fps_orig:.2f}x\")\n",
    "\n",
    "print(\"Left: Original, Right: RCCA\")\n",
    "display(make_image_grid([img_orig, img_rcca], 1, 2))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "semdraw",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open('notebooks/demo_comparison_rcca.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
