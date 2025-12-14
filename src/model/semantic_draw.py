# Copyright (c) 2025 Jaerin Lee

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from diffusers import DiffusionPipeline, LCMScheduler, EulerDiscreteScheduler, AutoencoderTiny
from huggingface_hub import hf_hub_download

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange

import random
from collections import deque
from typing import Tuple, List, Literal, Optional, Union
from PIL import Image

from util import load_model, gaussian_lowpass, shift_to_mask_bbox_center
from data import BackgroundObject, LayerObject, BackgroundState #, LayerState
from .rcca import RegionConstrainedAttnProcessor
import math


class SemanticDraw(nn.Module):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        sd_version: Literal['1.5'] = '1.5',
        hf_key: Optional[str] = None,
        lora_key: Optional[str] = None,
        use_tiny_vae: bool = True,
        t_index_list: List[int] = [0, 4, 12, 25, 37], # [0, 5, 16, 18, 20, 37], Magic number.
        width: int = 512,
        height: int = 512,
        frame_buffer_size: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        cfg_type: Literal['none', 'full', 'self', 'initialize'] = 'none',
        seed: int = 2024,
        seedfix: bool = False,
        autoflush: bool = True,
        default_mask_std: float = 8.0,
        default_mask_strength: float = 1.0,
        default_prompt_strength: float = 0.95,
        bootstrap_steps: int = 1,
        bootstrap_mix_steps: float = 1.0,
        # bootstrap_leak_sensitivity: float = 0.2,
        preprocess_mask_cover_alpha: float = 0.3, # TODO
        prompt_queue_capacity: int = 256,
        mask_type: Literal['discrete', 'semi-continuous', 'continuous'] = 'continuous',
        use_xformers: bool = True,
    ) -> None:
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.seedfix = seedfix
        self.sd_version = sd_version

        self.autoflush = autoflush
        self.default_mask_std = default_mask_std
        self.default_mask_strength = default_mask_strength
        self.default_prompt_strength = default_prompt_strength
        self.bootstrap_steps = (
            bootstrap_steps > torch.arange(len(t_index_list))).to(dtype=self.dtype, device=self.device)
        self.bootstrap_mix_steps = bootstrap_mix_steps
        self.bootstrap_mix_ratios = (
            bootstrap_mix_steps - torch.arange(len(t_index_list), dtype=self.dtype, device=self.device)).clip_(0, 1)
        # self.bootstrap_leak_sensitivity = bootstrap_leak_sensitivity
        self.preprocess_mask_cover_alpha = preprocess_mask_cover_alpha
        self.mask_type = mask_type

        ### State definition

        # [0. Start]                 -(prepare)->           [1. Initialized]
        # [1. Initialized]           -(update_background)-> [2. Background Registered] (len(self.prompts)==0)
        # [2. Background Registered] -(update_layers)->     [3. Unflushed] (len(self.prompts)>0)

        # [3. Unflushed]             -(flush)->             [4. Ready]
        # [4. Ready]                 -(any updates)->       [3. Unflushed]
        # [4. Ready]                 -(__call__)->          [4. Ready], continuously returns generated image.

        self.ready_checklist = {
            'initialized': False,
            'background_registered': False,
            'layers_ready': False,
            'flushed': False,
        }

        ### Session state update queue: for lazy update policy for streaming applications.

        self.update_buffer = {
            'background': None,                            # Maintains a single instance of BackgroundObject
            'layers': deque(maxlen=prompt_queue_capacity), # Maintains a queue of LayerObjects
        }

        print(f'[INFO]     Loading Stable Diffusion...')
        get_scheduler = lambda pipe: LCMScheduler.from_config(pipe.scheduler.config)
        lora_weight_name = None
        if self.sd_version == '1.5':
            if hf_key is not None:
                print(f'[INFO]     Using custom model key: {hf_key}')
                model_key = hf_key
            else:
                model_key = 'runwayml/stable-diffusion-v1-5'
            lora_key = 'latent-consistency/lcm-lora-sdv1-5'
            lora_weight_name = 'pytorch_lora_weights.safetensors'
        # elif self.sd_version == 'xl':
        #     model_key = 'stabilityai/stable-diffusion-xl-base-1.0'
        #     lora_key = 'latent-consistency/lcm-lora-sdxl'
        #     lora_weight_name = 'pytorch_lora_weights.safetensors'
        else:
            raise ValueError(f'Stable Diffusion version {self.sd_version} not supported.')

        ### Internally stored "Session" states

        self.state = {
            'background': BackgroundState(), # Maintains a single instance of BackgroundState
            # 'layers': LayerState(),          # Maintains a single instance of LayerState
            'model_key': model_key,          # The Hugging Face model ID.
        }

        # Create model
        self.i2t_processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b')
        self.i2t_model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')

        self.pipe = load_model(model_key, self.sd_version, self.device, self.dtype)

        self.pipe.load_lora_weights(lora_key, weight_name=lora_weight_name, adapter_name='lcm')
        self.pipe.fuse_lora(
            fuse_unet=True,
            fuse_text_encoder=True,
            lora_scale=1.0,
            safe_fusing=False,
        )
        if use_xformers:
            self.pipe.enable_xformers_memory_efficient_attention()

        self.vae = (
            AutoencoderTiny.from_pretrained('madebyollin/taesd').to(device=self.device, dtype=self.dtype)
            if use_tiny_vae else self.pipe.vae
        )
        # self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet
        self.vae_scale_factor = self.pipe.vae_scale_factor

        self.scheduler = get_scheduler(self.pipe)
        self.scheduler.set_timesteps(num_inference_steps)

        self.generator = None

        # Lock the canvas size--changing the canvas size can be implemented by reloading the module.
        self.height = height
        self.width = width
        self.latent_height = int(height // self.pipe.vae_scale_factor)
        self.latent_width = int(width // self.pipe.vae_scale_factor)

        # For bootstrapping.
        self.white = self.encode_imgs(torch.ones(1, 3, height, width, dtype=self.dtype, device=self.device))

        # StreamDiffusion setting.
        self.t_list = t_index_list
        assert len(self.t_list) > 1, 'Current version only supports diffusion models with multiple steps.'
        self.frame_bff_size = frame_buffer_size  # f
        self.denoising_steps_num = len(self.t_list)  # t=2
        self.cfg_type = cfg_type
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = 1.0 if self.cfg_type == 'none' else guidance_scale
        self.delta = delta

        self.batch_size = self.denoising_steps_num * frame_buffer_size  # T = t*f
        if self.cfg_type == 'initialize':
            self.trt_unet_batch_size = (self.denoising_steps_num + 1) * self.frame_bff_size
        elif self.cfg_type == 'full':
            self.trt_unet_batch_size = 2 * self.denoising_steps_num * self.frame_bff_size
        else:
            self.trt_unet_batch_size = self.denoising_steps_num * frame_buffer_size

        print(f'[INFO]     Model is loaded!')

        # Set up RCCA Custom Attention Processors
        attn_procs = {}
        for name, _ in self.unet.attn_processors.items():
            # Check if it's a cross-attention layer (usually has 'attn2')
            # Assuming standard Stable Diffusion U-Net naming
            if name.endswith("attn2.processor"):
                 attn_procs[name] = RegionConstrainedAttnProcessor()
            else:
                 # Keep default for self-attention
                 attn_procs[name] = self.unet.attn_processors[name]
        
        self.unet.set_attn_processor(attn_procs)
        print(f'[INFO]     RCCA Attention Processors initialized!')

        self.reset_seed(self.generator, seed)
        self.reset_latent()
        self.prepare()

        print(f'[INFO]     Parameters prepared!')

        self.ready_checklist['initialized'] = True

    @property
    def background(self) -> BackgroundState:
        return self.state['background']

    # @property
    # def layers(self) -> LayerState:
    #     return self.state['layers']

    @property
    def num_layers(self) -> int:
        return len(self.prompts) if hasattr(self, 'prompts') else 0

    @property
    def is_ready_except_flush(self) -> bool:
        return all(v for k, v in self.ready_checklist.items() if k != 'flushed')

    @property
    def is_flush_needed(self) -> bool:
        return self.autoflush and not self.ready_checklist['flushed']

    @property
    def is_ready(self) -> bool:
        return self.is_ready_except_flush and not self.is_flush_needed

    @property
    def is_dirty(self) -> bool:
        return not (self.update_buffer['background'] is None and len(self.update_buffer['layers']) == 0)

    @property
    def has_background(self) -> bool:
        return self.background.is_empty

    # @property
    # def has_layers(self) -> bool:
    #     return len(self.layers) > 0

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}(\n\tbackground: {str(self.background)},\n\t'
            f'model_key: {self.state["model_key"]}\n)'
            # f'layers: {str(self.layers)},\n\tmodel_key: {self.state["model_key"]}\n)'
        )

    def check_integrity(self, throw_error: bool = True) -> bool:
        p = len(self.prompts)
        flag = (
            p != len(self.negative_prompts) or
            p != len(self.prompt_strengths) or
            p != len(self.masks) or
            p != len(self.mask_strengths) or
            p != len(self.mask_stds) or
            p != len(self.original_masks)
        )
        if flag and throw_error:
            print(
                f'LayerState(\n\tlen(prompts): {p},\n\tlen(negative_prompts): {len(self.negative_prompts)},\n\t'
                f'len(prompt_strengths): {len(self.prompt_strengths)},\n\tlen(masks): {len(self.masks)},\n\t'
                f'len(mask_stds): {len(self.mask_stds)},\n\tlen(mask_strengths): {len(self.mask_strengths)},\n\t'
                f'len(original_masks): {len(self.original_masks)}\n)'
            )
            raise ValueError('[ERROR]    LayerState is corrupted!')
        return not flag

    def check_ready(self) -> bool:
        all_except_flushed = all(v for k, v in self.ready_checklist.items() if k != 'flushed')
        if all_except_flushed:
            if self.is_flush_needed:
                self.flush()
            return True

        print('[WARNING]  MagicDraw module is not ready yet! Complete the checklist:')
        for k, v in self.ready_checklist.items():
            prefix = '  [ v ] ' if v else '  [ x ] '
            print(prefix + k.replace('_', ' '))
        return False

    def reset_seed(self, generator: Optional[torch.Generator] = None, seed: Optional[int] = None) -> None:
        generator = torch.Generator(self.device) if generator is None else generator
        seed = self.seed if seed is None else seed
        self.generator = generator
        self.generator.manual_seed(seed)

        self.init_noise = torch.randn((self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator, device=self.device, dtype=self.dtype)
        self.stock_noise = torch.zeros_like(self.init_noise)

        self.ready_checklist['flushed'] = False

    def reset_latent(self) -> None:
        # initialize x_t_latent (it can be any random tensor)
        b = (self.denoising_steps_num - 1) * self.frame_bff_size
        self.x_t_latent_buffer = torch.zeros(
            (b, 4, self.latent_height, self.latent_width), dtype=self.dtype, device=self.device)

    def reset_state(self) -> None:
        # TODO Reset states for context switch between multiple users.
        pass

    def prepare(self) -> None:
        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.timesteps = self.scheduler.timesteps.to(self.device)
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])
        sub_timesteps_tensor = torch.tensor(self.sub_timesteps, dtype=torch.long, device=self.device)
        self.sub_timesteps_tensor = sub_timesteps_tensor.repeat_interleave(self.frame_bff_size, dim=0)

        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)
        self.c_skip = torch.stack(c_skip_list).view(len(self.t_list), 1, 1, 1).to(dtype=self.dtype, device=self.device)
        self.c_out = torch.stack(c_out_list).view(len(self.t_list), 1, 1, 1).to(dtype=self.dtype, device=self.device)

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        alpha_prod_t_sqrt = (torch.stack(alpha_prod_t_sqrt_list).view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device))
        beta_prod_t_sqrt = (torch.stack(beta_prod_t_sqrt_list).view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device))
        self.alpha_prod_t_sqrt = alpha_prod_t_sqrt.repeat_interleave(self.frame_bff_size, dim=0)
        self.beta_prod_t_sqrt = beta_prod_t_sqrt.repeat_interleave(self.frame_bff_size, dim=0)

        noise_lvs = ((1 - self.scheduler.alphas_cumprod.to(self.device)[self.sub_timesteps_tensor]) ** 0.5)
        self.noise_lvs = noise_lvs[None, :, None, None, None]
        self.next_noise_lvs = torch.cat([noise_lvs[1:], noise_lvs.new_zeros(1)])[None, :, None, None, None]

    @torch.no_grad()
    def get_text_prompts(self, image: Image.Image) -> str:
        r"""A convenient method to extract text prompt from an image.

        This is called if the user does not provide background prompt but only
        the background image. We use BLIP-2 to automatically generate prompts.

        Args:
            image (Image.Image): A PIL image.

        Returns:
            A single string of text prompt.
        """
        if not hasattr(self, 'i2t_model') or self.i2t_model is None:
            print("[WARN] BLIP-2 model not loaded. Returning empty prompt.")
            return ""

        question = 'Question: What are in the image? Answer:'
        inputs = self.i2t_processor(image, question, return_tensors='pt')
        if hasattr(self.i2t_model, 'device') and self.i2t_model.device.type != 'cpu':
            inputs = {k: v.to(self.i2t_model.device) for k, v in inputs.items()}
        out = self.i2t_model.generate(**inputs, max_new_tokens=77)
        prompt = self.i2t_processor.decode(out[0], skip_special_tokens=True).strip()
        return prompt

    @torch.no_grad()
    def encode_imgs(
        self,
        imgs: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        add_noise: bool = False,
    ) -> torch.Tensor:
        r"""A wrapper function for VAE encoder of the latent diffusion model.

        Args:
            imgs (torch.Tensor): An image to get StableDiffusion latents.
                Expected shape: (B, 3, H, W). Expected pixel scale: [0, 1].
            generator (Optional[torch.Generator]): Seed for KL-Autoencoder.
            add_noise (bool): Turn this on for a noisy latent.

        Returns:
            An image latent embedding with 1/8 size (depending on the auto-
            encoder. Shape: (B, 4, H//8, W//8).
        """
        def _retrieve_latents(
            encoder_output: torch.Tensor,
            generator: Optional[torch.Generator] = None,
            sample_mode: str = 'sample',
        ):
            if hasattr(encoder_output, 'latent_dist') and sample_mode == 'sample':
                return encoder_output.latent_dist.sample(generator)
            elif hasattr(encoder_output, 'latent_dist') and sample_mode == 'argmax':
                return encoder_output.latent_dist.mode()
            elif hasattr(encoder_output, 'latents'):
                return encoder_output.latents
            else:
                raise AttributeError('[ERROR]    Could not access latents of provided encoder_output')

        imgs = 2 * imgs - 1
        latents = self.vae.config.scaling_factor * _retrieve_latents(self.vae.encode(imgs), generator=generator)
        if add_noise:
            latents = self.alpha_prod_t_sqrt[0] * latents + self.beta_prod_t_sqrt[0] * self.init_noise[0]
        return latents

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        r"""A wrapper function for VAE decoder of the latent diffusion model.

        Args:
            latents (torch.Tensor): An image latent to get associated images.
                Expected shape: (B, 4, H//8, W//8).

        Returns:
            An image latent embedding with 1/8 size (depending on the auto-
            encoder. Shape: (B, 3, H, W).
        """
        latents = 1 / self.vae.config.scaling_factor * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clip_(0, 1)
        return imgs

    @torch.no_grad()
    def update_background(
        self,
        image: Optional[Image.Image] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
    ) -> bool:
        flag_changed = False
        if image is not None:
            image_ = image.resize((self.width, self.height))
            prompt = self.get_text_prompts(image_) if prompt is None else prompt
            negative_prompt = '' if negative_prompt is None else negative_prompt
            embed = self.pipe.encode_prompt(
                prompt=[prompt],
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=(self.guidance_scale > 1.0),
                negative_prompt=[negative_prompt],
            )  # ((1, 77, 768): cond, (1, 77, 768): uncond)

            self.state['background'].image = image
            self.state['background'].latent = (
                self.encode_imgs(T.ToTensor()(image_)[None].to(self.device, self.dtype))
            )  # (1, 3, H, W)
            self.state['background'].prompt = prompt
            self.state['background'].negative_prompt = negative_prompt
            self.state['background'].embed = embed

            if self.bootstrap_steps[0] > 0:
                mix_ratio = self.bootstrap_mix_ratios[:, None, None, None]
                self.bootstrap_latent = mix_ratio * self.white + (1.0 - mix_ratio) * self.state['background'].latent

            self.ready_checklist['background_registered'] = True
            flag_changed = True
        else:
            if not self.ready_checklist['background_registered']:
                print('[WARNING]  Register background image first! Request ignored.')
                return False

            if prompt is not None:
                self.background.prompt = prompt
                flag_changed = True
            if negative_prompt is not None:
                self.background.negative_prompt = negative_prompt
                flag_changed = True
            if flag_changed:
                self.background.embed = self.pipe.encode_prompt(
                    prompt=[self.background.prompt],
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=(self.guidance_scale > 1.0),
                    negative_prompt=[self.background.negative_prompt],
                )  # ((1, 77, 768): cond, (1, 77, 768): uncond)
    
        self.ready_checklist['flushed'] = not flag_changed
        return flag_changed

    @torch.no_grad()
    def process_mask(
        self,
        masks: Optional[Union[torch.Tensor, Image.Image, List[Image.Image]]] = None,
        strength: Optional[Union[torch.Tensor, float]] = None,
        std: Optional[Union[torch.Tensor, float]] = None,
    ) -> Tuple[torch.Tensor]:
        r"""Fast preprocess of masks for region-based generation with fine-
        grained controls.

        Mask preprocessing is done in four steps:
         1. Resizing: Resize the masks into the specified width and height by
            nearest neighbor interpolation.
         2. (Optional) Ordering: Masks with higher indices are considered to
            cover the masks with smaller indices. Covered masks are decayed
            in its alpha value by the specified factor of
            `preprocess_mask_cover_alpha`.
         3. Blurring: Gaussian blur is applied to the mask with the specified
            standard deviation (isotropic). This results in gradual increase of
            masked region as the timesteps evolve, naturally blending fore-
            ground and the predesignated background. Not strictly required if
            you want to produce images from scratch withoout background.
         4. Quantization: Split the real-numbered masks of value between [0, 1]
            into predefined noise levels for each quantized scheduling step of
            the diffusion sampler. For example, if the diffusion model sampler
            has noise level of [0.9977, 0.9912, 0.9735, 0.8499, 0.5840], which
            is the default noise level of this module with schedule [0, 4, 12,
            25, 37], the masks are split into binary masks whose values are
            greater than these levels. This results in tradual increase of mask
            region as the timesteps increase. Details are described in our
            paper.

        On the Three Modes of `mask_type`:
            `self.mask_type` is predefined at the initialization stage of this
            pipeline. Three possible modes are available: 'discrete', 'semi-
            continuous', and 'continuous'. These define the mask quantization
            modes we use. Basically, this (subtly) controls the smoothness of
            foreground-background blending. Continuous modes produces nonbinary
            masks to further blend foreground and background latents by linear-
            ly interpolating between them. Semi-continuous masks only applies
            continuous mask at the last step of the LCM sampler. Due to the
            large step size of the LCM scheduler, we find that our continuous
            blending helps generating seamless inpainting and editing results.

        Args:
            masks (Union[torch.Tensor, Image.Image, List[Image.Image]]): Masks.
            strength (Optional[Union[torch.Tensor, float]]): Mask strength that
                overrides the default value. A globally multiplied factor to
                the mask at the initial stage of processing. Can be applied
                seperately for each mask.
            std (Optional[Union[torch.Tensor, float]]): Mask blurring Gaussian
                kernel's standard deviation. Overrides the default value. Can
                be applied seperately for each mask.

        Returns: A tuple of tensors.
          - masks: Preprocessed (ordered, blurred, and quantized) binary/non-
                binary masks (see the explanation on `mask_type` above) for
                region-based image synthesis.
          - strengths: Return mask strengths for caching.
          - std: Return mask blur standard deviations for caching.
          - original_masks: Return original masks for caching.
        """
        if masks is None:
            kwargs = {'dtype': self.dtype, 'device': self.device}
            original_masks = torch.zeros((0, 1, self.latent_height, self.latent_width), dtype=self.dtype)
            masks = torch.zeros((0, self.batch_size, 1, self.latent_height, self.latent_width), **kwargs)
            strength = torch.zeros((0,), **kwargs)
            std = torch.zeros((0,), **kwargs)
            return masks, strength, std, original_masks

        if isinstance(masks, Image.Image):
            masks = [masks]
        if isinstance(masks, (tuple, list)):
            # Assumes white background for Image.Image;
            # inverted boolean masks with shape (1, 1, H, W) for torch.Tensor.
            masks = torch.cat([
                # (T.ToTensor()(mask.resize((self.width, self.height), Image.NEAREST)) < 0.5)[None, :1]
                (1.0 - T.ToTensor()(mask.resize((self.width, self.height), Image.BILINEAR)))[None, :1]
                for mask in masks
            ], dim=0).float().clip_(0, 1)
        original_masks = masks            
        masks = masks.float().to(self.device)

        # Background mask alpha is decayed by the specified factor where foreground masks covers it.
        if self.preprocess_mask_cover_alpha > 0:
            masks = torch.stack([
                torch.where(
                    masks[i + 1:].sum(dim=0) > 0,
                    mask * self.preprocess_mask_cover_alpha,
                    mask,
                ) if i < len(masks) - 1 else mask
                for i, mask in enumerate(masks)
            ], dim=0)

        if std is None:
            std = self.default_mask_std
        if isinstance(std, (int, float)):
            std = [std] * len(masks)
        if isinstance(std, (list, tuple)):
            std = torch.as_tensor(std, dtype=torch.float, device=self.device)

        # Mask preprocessing parameters are fetched from the default settings.
        if strength is None:
            strength = self.default_mask_strength
        if isinstance(strength, (int, float)):
            strength = [strength] * len(masks)
        if isinstance(strength, (list, tuple)):
            strength = torch.as_tensor(strength, dtype=torch.float, device=self.device)

        if (std > 0).any():
            std = torch.where(std > 0, std, 1e-5)
            masks = gaussian_lowpass(masks, std)
        # NOTE: This `strength` aligns with `denoising strength`. However, with LCM, using strength < 0.96
        #       gives unpleasant results.
        masks = masks * strength[:, None, None, None]
        masks = masks.unsqueeze(1).repeat(1, self.noise_lvs.shape[1], 1, 1, 1)

        if self.mask_type == 'discrete':
            # Discrete mode.
            masks = masks > self.noise_lvs
        elif self.mask_type == 'semi-continuous':
            # Semi-continuous mode (continuous at the last step only).
            masks = torch.cat((
                masks[:, :-1] > self.noise_lvs[:, :-1],
                (
                    (masks[:, -1:] - self.next_noise_lvs[:, -1:])
                    / (self.noise_lvs[:, -1:] - self.next_noise_lvs[:, -1:])
                ).clip_(0, 1),
            ), dim=1)
        elif self.mask_type == 'continuous':
            # Continuous mode: Have the exact same `1` coverage with discrete mode, but the mask gradually
            #                  decreases continuously after the discrete mode boundary to become `0` at the
            #                  next lower threshold.
            masks = ((masks - self.next_noise_lvs) / (self.noise_lvs - self.next_noise_lvs)).clip_(0, 1)

        # NOTE: Post processing mask strength does not align with conventional 'denoising_strength'. However,
        #       fine-grained mask alpha channel tuning is available with this form.
        # masks = masks * strength[None, :, None, None, None]

        masks = rearrange(masks.float(), 'p t () h w -> (p t) () h w')
        masks = F.interpolate(masks, size=(self.latent_height, self.latent_width), mode='nearest')
        masks = rearrange(masks.to(self.dtype), '(p t) () h w -> p t () h w', p=len(std))
        return masks, strength, std, original_masks

    @torch.no_grad()
    def update_layers(
        self,
        prompts: Union[str, List[str]],
        negative_prompts: Optional[Union[str, List[str]]] = None,
        suffix: Optional[str] = None, #', background is ',
        prompt_strengths: Optional[Union[torch.Tensor, float, List[float]]] = None,
        masks: Optional[Union[torch.Tensor, Image.Image, List[Image.Image]]] = None,
        mask_strengths: Optional[Union[torch.Tensor, float, List[float]]] = None,
        mask_stds: Optional[Union[torch.Tensor, float, List[float]]] = None,
    ) -> None:
        if not self.ready_checklist['background_registered']:
            print('[WARNING]  Register background image first! Request ignored.')
            return

        ### Register prompts

        if isinstance(prompts, str):
            prompts = [prompts]
        if negative_prompts is None:
            negative_prompts = ''
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        fg_prompt = [p + suffix + self.background.prompt if suffix is not None else p for p in prompts]
        self.prompts = fg_prompt
        self.negative_prompts = negative_prompts
        p = self.num_layers

        e = self.pipe.encode_prompt(
            prompt=fg_prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=(self.guidance_scale > 1.0),
            negative_prompt=negative_prompts,
        )  # (p, 77, 768)

        if prompt_strengths is None:
            prompt_strengths = self.default_prompt_strength
        if isinstance(prompt_strengths, (int, float)):
            prompt_strengths = [prompt_strengths] * p
        if isinstance(prompt_strengths, (list, tuple)):
            prompt_strengths = torch.as_tensor(prompt_strengths, dtype=self.dtype, device=self.device)
        self.prompt_strengths = prompt_strengths

        s = prompt_strengths[:, None, None]
        self.prompt_embeds = torch.lerp(self.background.embed[0], e[0], s).repeat(self.batch_size, 1, 1)  # (T * p, 77, 768)
        if self.guidance_scale > 1.0 and self.cfg_type in ('initialize', 'full'):
            b = self.batch_size if self.cfg_type == 'full' else self.frame_bff_size
            uncond_prompt_embeds = torch.lerp(self.background.embed[1], e[1], s).repeat(b, 1, 1)  # (T * p, 77, 768)
            self.prompt_embeds = torch.cat([uncond_prompt_embeds, self.prompt_embeds], dim=0)  # (2 * T * p, 77, 768)

        self.sub_timesteps_tensor_ = self.sub_timesteps_tensor.repeat_interleave(p)  # (T * p,)
        self.init_noise_ = self.init_noise.repeat_interleave(p, dim=0)  # (T * p, 77, 768)
        self.stock_noise_ = self.stock_noise.repeat_interleave(p, dim=0)  # (T * p, 77, 768)
        self.c_out_ = self.c_out.repeat_interleave(p, dim=0)  # (T * p, 1, 1, 1)
        self.c_skip_ = self.c_skip.repeat_interleave(p, dim=0)  # (T * p, 1, 1, 1)
        self.beta_prod_t_sqrt_ = self.beta_prod_t_sqrt.repeat_interleave(p, dim=0)  # (T * p, 1, 1, 1)
        self.alpha_prod_t_sqrt_ = self.alpha_prod_t_sqrt.repeat_interleave(p, dim=0)  # (T * p, 1, 1, 1)

        ### Register new masks

        if isinstance(masks, Image.Image):
            masks = [masks]
        n = len(masks) if masks is not None else 0

        # Modificiation.
        masks, mask_strengths, mask_stds, original_masks = self.process_mask(masks, mask_strengths, mask_stds)

        self.counts = masks.sum(dim=0)  # (T, 1, h, w)
        self.bg_mask = (1 - self.counts).clip_(0, 1)  # (T, 1, h, w)
        self.masks = masks  # (p, T, 1, h, w)
        self.mask_strengths = mask_strengths  # (p,)
        self.mask_stds = mask_stds  # (p,)
        self.original_masks = original_masks  # (p, 1, h, w)

        if p > n:
            # Add more masks: counts and bg_masks are not changed, but only masks are changed.
            self.masks = torch.cat((
                self.masks,
                torch.zeros(
                    (p - n, self.batch_size, 1, self.latent_height, self.latent_width),
                    dtype=self.dtype,
                    device=self.device,
                ),
            ), dim=0)
            print(f'[WARNING]  Detected more prompts ({p}) than masks ({n}). '
                  'Automatically adds blank masks for the additional prompts.')
        elif p < n:
            # Warns user to add more prompts.
            print(f'[WARNING]  Detected more masks ({n}) than prompts ({p}). '
                  'Additional masks are ignored until more prompts are provided.')

        self.ready_checklist['layers_ready'] = True
        self.ready_checklist['flushed'] = False

    @torch.no_grad()
    def update_masks(
        self,
        masks: Optional[Union[torch.Tensor, Image.Image, List[Image.Image]]] = None,
        mask_strengths: Optional[Union[torch.Tensor, float, List[float]]] = None,
        mask_stds: Optional[Union[torch.Tensor, float, List[float]]] = None,
    ) -> None:
        if not self.ready_checklist['background_registered']:
            print('[WARNING]  Register background image first! Request ignored.')
            return

        ### Register new masks

        if isinstance(masks, Image.Image):
            masks = [masks]
        p = self.num_layers
        n = len(masks) if masks is not None else 0

        # Modificiation.
        masks, mask_strengths, mask_stds, original_masks = self.process_mask(masks, mask_strengths, mask_stds)

        self.counts = masks.sum(dim=0)  # (T, 1, h, w)
        self.bg_mask = (1 - self.counts).clip_(0, 1)  # (T, 1, h, w)
        self.masks = masks  # (p, T, 1, h, w)
        self.mask_strengths = mask_strengths  # (p,)
        self.mask_stds = mask_stds  # (p,)
        self.original_masks = original_masks  # (p, 1, h, w)

        if p > n:
            # Add more masks: counts and bg_masks are not changed, but only masks are changed.
            self.masks = torch.cat((
                self.masks,
                torch.zeros(
                    (p - n, self.batch_size, 1, self.latent_height, self.latent_width),
                    dtype=self.dtype,
                    device=self.device,
                ),
            ), dim=0)
            print(f'[WARNING]  Detected more prompts ({p}) than masks ({n}). '
                  'Automatically adds blank masks for the additional prompts.')
        elif p < n:
            # Warns user to add more prompts.
            print(f'[WARNING]  Detected more masks ({n}) than prompts ({p}). '
                  'Additional masks are ignored until more prompts are provided.')

    @torch.no_grad()
    def update_single_layer(
        self,
        idx: Optional[int] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        suffix: Optional[str] = None, #', background is ',
        prompt_strength: Optional[float] = None,
        mask: Optional[Union[torch.Tensor, Image.Image]] = None,
        mask_strength: Optional[float] = None,
        mask_std: Optional[float] = None,
    ) -> None:

        ### Possible input combinations and expected behaviors

        # The module will consider a layer, a pair of (prompt, mask), to be 'active' only if a prompt
        # is registered. A blank mask will be assigned if no mask is provided for the 'active' layer.
        # The layers should be in either of ('active', 'inactive') states. 'inactive' layers will not
        # receive any input unless equipped with prompt. 'active' layers receive any input and modify
        # their states accordingly. In the actual implementation, only the 'active' layers are stored
        # and can be accessed by the fields. Values len(self.prompts) = self.num_layers is the number
        # of 'active' layers.

        # If no background is registered. The layers should be all 'inactive'.
        if not self.ready_checklist['background_registered']:
            print('[WARNING]  Register background image first! Request ignored.')
            return

        # The first layer create request should be carrying a prompt. If only mask is drawn without a
        # prompt, it just ignores the request--the user will update her request soon.
        if self.num_layers == 0:
            if prompt is not None:
                self.update_layers(
                    prompts=prompt,
                    negative_prompts=negative_prompt,
                    suffix=suffix,
                    prompt_strengths=prompt_strength,
                    masks=mask,
                    mask_strengths=mask_strength,
                    mask_stds=mask_std,
                )
            return

        # Invalid request indices -> considered as a layer add request.
        if idx is None or idx > self.num_layers or idx < 0:
            idx = self.num_layers

        # Two modes for the layer edits: 'append mode' and 'edit mode'. 'append mode' appends a new
        # layer at the end of the layers list. 'edit mode' modifies internal variables for the given
        # index. 'append mode' is defined by the request index and strictly requires a prompt input.
        is_appending = idx == self.num_layers
        if is_appending and prompt is None:
            print(f'[WARNING]  Creating a new prompt at index ({idx}) but found no prompt. Request ignored.')
            return

        ### Register prompts

        # | prompt    | neg_prompt | append mode (idx==len)  | edit mode (idx<len)  |
        # | --------- | ---------- | ----------------------- | -------------------- |
        # | given     | given      | append new prompt embed | replace prompt embed |
        # | given     | not given  | append new prompt embed | replace prompt embed |
        # | not given | given      | NOT ALLOWED             | replace prompt embed |
        # | not given | not given  | NOT ALLOWED             | do nothing           |

        # | prompt_strength | append mode (idx==len) | edit mode (idx<len)                            |
        # | --------------- | ---------------------- | ---------------------------------------------- |
        # | given           | use given strength     | use given strength                             |
        # | not given       | use default strength   | replace strength / if no existing, use default |

        p = self.num_layers

        flag_prompt_edited = (
            prompt is not None or
            negative_prompt is not None or
            prompt_strength is not None
        )

        if flag_prompt_edited:
            is_double_cond = self.guidance_scale > 1.0 and self.cfg_type in ('initialize', 'full')

            # Synchonize the internal state.

            # We have asserted that prompt is not None if the mode is 'appending'.
            if prompt is not None:
                if suffix is not None:
                    prompt = prompt + suffix + self.background.prompt
                if is_appending:
                    self.prompts.append(prompt)
                else:
                    self.prompts[idx] = prompt

            if negative_prompt is not None:
                if is_appending:
                    self.negative_prompts.append(negative_prompt)
                else:
                    self.negative_prompts[idx] = negative_prompt
            elif is_appending:
                # Make sure that negative prompts are well specified.
                self.negative_prompts.append('')

            if is_appending:
                if prompt_strength is None:
                    prompt_strength = self.default_prompt_strength
                self.prompt_strengths = torch.cat((
                    self.prompt_strengths,
                    torch.as_tensor([prompt_strength], dtype=self.dtype, device=self.device),
                ), dim=0)
            elif prompt_strength is not None:
                self.prompt_strengths[idx] = prompt_strength

            # Edit currently stored prompt embeddings.

            if is_double_cond:
                uncond_prompt_embed_, prompt_embed_ = torch.chunk(self.prompt_embeds, 2, dim=0)
                uncond_prompt_embed_ = rearrange(uncond_prompt_embed_, '(t p) c1 c2 -> t p c1 c2', p=p)
                prompt_embed_ = rearrange(prompt_embed_, '(t p) c1 c2 -> t p c1 c2', p=p)
            else:
                uncond_prompt_embed_ = None
                prompt_embed_ = rearrange(self.prompt_embeds, '(t p) c1 c2 -> t p c1 c2', p=p)

            e = self.pipe.encode_prompt(
                prompt=self.prompts[idx],
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=(self.guidance_scale > 1.0),
                negative_prompt=self.negative_prompts[idx],
            )  # (1, 77, 768), (1, 77, 768)

            s = self.prompt_strengths[idx]
            t = prompt_embed_.shape[0]
            prompt_embed = torch.lerp(self.background.embed[0], e[0], s)[None].repeat(t, 1, 1, 1)  # (1, 77, 768)
            if is_double_cond:
                uncond_prompt_embed = torch.lerp(self.background.embed[1], e[1], s)[None].repeat(t, 1, 1, 1)  # (1, 77, 768)

            if is_appending:
                prompt_embed_ = torch.cat((prompt_embed_, prompt_embed), dim=1)
                if is_double_cond:
                    uncond_prompt_embed_ = torch.cat((uncond_prompt_embed_, uncond_prompt_embed), dim=1)
            else:
                prompt_embed_[:, idx:(idx + 1)] = prompt_embed
                if is_double_cond:
                    uncond_prompt_embed_[:, idx:(idx + 1)] = uncond_prompt_embed

            self.prompt_embeds = rearrange(prompt_embed_, 't p c1 c2 -> (t p) c1 c2')
            if is_double_cond:
                uncond_prompt_embeds = rearrange(uncond_prompt_embed_, 't p c1 c2 -> (t p) c1 c2')
                self.prompt_embeds = torch.cat([uncond_prompt_embeds, self.prompt_embeds], dim=0)  # (2 * T * p, 77, 768)

            self.ready_checklist['flushed'] = False

        if is_appending:
            p = self.num_layers
            self.sub_timesteps_tensor_ = self.sub_timesteps_tensor.repeat_interleave(p)  # (T * p,)
            self.init_noise_ = self.init_noise.repeat_interleave(p, dim=0)  # (T * p, 77, 768)
            self.stock_noise_ = self.stock_noise.repeat_interleave(p, dim=0)  # (T * p, 77, 768)
            self.c_out_ = self.c_out.repeat_interleave(p, dim=0)  # (T * p, 1, 1, 1)
            self.c_skip_ = self.c_skip.repeat_interleave(p, dim=0)  # (T * p, 1, 1, 1)
            self.beta_prod_t_sqrt_ = self.beta_prod_t_sqrt.repeat_interleave(p, dim=0)  # (T * p, 1, 1, 1)
            self.alpha_prod_t_sqrt_ = self.alpha_prod_t_sqrt.repeat_interleave(p, dim=0)  # (T * p, 1, 1, 1)

        ### Register new masks

        # | mask      | std / str | append mode (idx==len)       | edit mode (idx<len)           |
        # | --------- | --------- | ---------------------------- | ----------------------------- |
        # | given     | given     | create mask with given val   | create mask with given val    |
        # | given     | not given | create mask with default val | create mask with existing val |
        # | not given | given     | create blank mask            | replace mask with given val   |
        # | not given | not given | create blank mask            | do nothing                    |

        flag_nonzero_mask = False
        if mask is not None:
            # Mask image is given -> create mask.
            mask, strength, std, original_mask = self.process_mask(mask, mask_strength, mask_std)
            flag_nonzero_mask = True

        elif is_appending:
            # No given mask & append mode  -> create white mask.
            mask = torch.zeros(
                (1, self.batch_size, 1, self.latent_height, self.latent_width),
                dtype=self.dtype,
                device=self.device,
            )
            strength = torch.as_tensor([self.default_mask_strength], dtype=self.dtype, device=self.device)
            std = torch.as_tensor([self.default_mask_std], dtype=self.dtype, device=self.device)
            original_mask = torch.zeros((1, 1, self.height, self.width), dtype=self.dtype, device=self.device)

        elif mask_std is not None or mask_strength is not None:
            # No given mask & edit mode & given std / str -> replace existing mask with given std / str.
            if mask_std is None:
                mask_std = self.mask_stds[idx:(idx + 1)]
            if mask_strength is None:
                mask_strength = self.mask_strengths[idx:(idx + 1)]
            mask, strength, std, original_mask = self.process_mask(
                self.original_masks[idx:(idx + 1)], mask_strength, mask_std)
            flag_nonzero_mask = True

        else:
            # No given mask & no given std & edit mode -> Do nothing.
            return

        if is_appending:
            # Append mode.
            self.masks = torch.cat((self.masks, mask), dim=0)  # (p, T, 1, h, w)
            self.mask_strengths = torch.cat((self.mask_strengths, strength), dim=0)  # (p,)
            self.mask_stds = torch.cat((self.mask_stds, std), dim=0)  # (p,)
            self.original_masks = torch.cat((self.original_masks, original_mask), dim=0)  # (p, 1, h, w)
            if flag_nonzero_mask:
                self.counts = self.counts + mask[0] if hasattr(self, 'counts') else mask[0]  # (T, 1, h, w)
                self.bg_mask = (1 - self.counts).clip_(0, 1)  # (T, 1, h, w)
        else:
            # Edit mode.
            if flag_nonzero_mask:
                self.counts = self.counts - self.masks[idx] + mask[0]  # (T, 1, h, w)
                self.bg_mask = (1 - self.counts).clip_(0, 1)  # (T, 1, h, w) 
            self.masks[idx:(idx + 1)] = mask  # (p, T, 1, h, w)
            self.mask_strengths[idx:(idx + 1)] = strength  # (p,)
            self.mask_stds[idx:(idx + 1)] = std  # (p,)
            self.original_masks[idx:(idx + 1)] = original_mask  # (p, 1, h, w)

        # if flag_nonzero_mask:
        #     self.ready_checklist['flushed'] = False

    @torch.no_grad()
    def register_all(
        self,
        prompts: Union[str, List[str]],
        masks: Union[Image.Image, List[Image.Image]],
        background: Image.Image,
        background_prompt: Optional[str] = None,
        background_negative_prompt: str = '',
        negative_prompts: Union[str, List[str]] = '',
        suffix: Optional[str] = None, #', background is ',
        prompt_strengths: float = 1.0,
        mask_strengths: float = 1.0,
        mask_stds: Union[torch.Tensor, float] = 10.0,
    ) -> None:
        # The order of this registration should not be changed!
        self.update_background(background, background_prompt, background_negative_prompt)
        self.update_layers(prompts, negative_prompts, suffix, prompt_strengths, masks, mask_strengths, mask_stds)

    def update(
        self,
        background: Optional[Image.Image] = None,
        background_prompt: Optional[str] = None,
        background_negative_prompt: Optional[str] = None,
        idx: Optional[int] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        suffix: Optional[str] = None,
        prompt_strength: Optional[float] = None,
        mask: Optional[Union[torch.Tensor, Image.Image]] = None,
        mask_strength: Optional[float] = None,
        mask_std: Optional[float] = None,
    ) -> None:
        # For lazy update (to solve minor synchonization problem with gradio).
        bq = BackgroundObject(
            image=background,
            prompt=background_prompt,
            negative_prompt=background_negative_prompt,
        )
        if not bq.is_empty:
            self.update_buffer['background'] = bq

        lq = LayerObject(
            idx=idx,
            prompt=prompt,
            negative_prompt=negative_prompt,
            suffix=suffix,
            prompt_strength=prompt_strength,
            mask=mask,
            mask_strength=mask_strength,
            mask_std=mask_std,
        )
        if not lq.is_empty:
            limit = self.update_buffer['layers'].maxlen

            # Optimize the prompt queue: Overrride uncommitted layers with the same idx.
            new_q = deque(maxlen=limit)
            for _ in range(len(self.update_buffer['layers'])):
                # Check from the newest to the oldest.
                # Copy old requests only if the current query does not carry those requests.
                query = self.update_buffer['layers'].pop()
                overriden = lq.merge(query)
                if not overriden:
                    new_q.appendleft(query)
            self.update_buffer['layers'] = new_q

            if len(self.update_buffer['layers']) == limit:
                print(f'[WARNING]  Maximum prompt change query limit ({limit}) is reached. '
                      f'Current query {lq} will be ignored.')
            else:
                self.update_buffer['layers'].append(lq)

    @torch.no_grad()
    def commit(self) -> None:
        flag_changed = self.is_dirty
        bq = self.update_buffer['background']
        lq = self.update_buffer['layers']
        count_bq_req = int(bq is not None and not bq.is_empty)
        count_lq_req = len(lq)

        if flag_changed:
            print(f'[INFO]     Requests found: {count_bq_req} background requests '
                  f'& {count_lq_req} layer requests:\n{str(bq)}, {", ".join([str(l) for l in lq])}')

        bq = self.update_buffer['background']
        if bq is not None:
            self.update_background(**vars(bq))
            self.update_buffer['background'] = None

        while len(lq) > 0:
            l = lq.popleft()
            self.update_single_layer(**vars(l))

        if flag_changed:
            print(f'[INFO]     Requests resolved: {count_bq_req} background requests '
                  f'& {count_lq_req} layer requests.')

    def scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        r"""Denoise-only step for reverse diffusion scheduler.

        Args:
            model_pred_batch (torch.Tensor): Noise prediction results.
            x_t_latent_batch (torch.Tensor): Noisy latent.
            idx (Optional[int]): Instead of timesteps (in [0, 1000]-scale) use
                indices for the timesteps tensor (ranged in
                [0, len(timesteps)-1]). Specify only if a single-index, not
                stream-batched inference is what you want.

        Returns:
            A denoised tensor with the same size as latent.
        """
        if idx is None:
            F_theta = (x_t_latent_batch - self.beta_prod_t_sqrt_ * model_pred_batch) / self.alpha_prod_t_sqrt_
            denoised_batch = self.c_out_ * F_theta + self.c_skip_ * x_t_latent_batch
        else:
            F_theta = (x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch) / self.alpha_prod_t_sqrt[idx]
            denoised_batch = self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch
        return denoised_batch

    def unet_step(
        self,
        x_t_latent: torch.Tensor,  # (T, 4, h, w)
        idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # RCCA Implementation: Single Forward Pass with Masked Attention
        
        p = self.num_layers
        # x_t_latent is (T, 4, h, w). We assume batch size 1 for T (stream batching usually means T distinct frames).
        # Actually T is the number of stream slots. Steps 0, 1, 2... 
        # x_t_latent shape is (T, 4, h, w). T is batch size for the U-Net.
        
        T_batch = x_t_latent.shape[0]
        
        # 1. Prepare Embeddings for Single Pass
        # self.prompt_embeds shape is (T*p, 77, 768) or (2*T*p, ...) if double cond.
        # We need to reshape to (T, p*77, 768).
        
        # Note: self.prompt_embeds was constructed for the "Batch splitting" method (repeating T for each p).
        # Structure: [T_step0_p0, T_step0_p1... T_stepN_pM] (interleaved? check update_layers)
        # update_layers: rearrange(prompt_embed_, 't p c1 c2 -> (t p) c1 c2')
        # So it is Interleaved: t0p0, t0p1, t0p2... t1p0...
        
        if self.guidance_scale > 1.0 and self.cfg_type in ('initialize', 'full'):
             # Structure: [Uncond(T*p), Cond(T*p)]
             # Split first
             all_embeds = self.prompt_embeds
             half = all_embeds.shape[0] // 2
             uncond_flat = all_embeds[:half] # (T*p, 77, 768)
             cond_flat = all_embeds[half:]   # (T*p, 77, 768)
             
             # Reshape to (T, p, 77, 768)
             uncond_reshaped = rearrange(uncond_flat, '(t p) l d -> t (p l) d', p=p, t=T_batch)
             cond_reshaped = rearrange(cond_flat, '(t p) l d -> t (p l) d', p=p, t=T_batch)
             
             # Concat for Classifier Free Guidance: [Uncond, Cond] in batch dim?
             # Standard diffusers: cat([uncond, cond]) -> Batch size * 2.
             # So we want (2*T, p*77, 768)
             text_embeds = torch.cat([uncond_reshaped, cond_reshaped], dim=0) # (2*T, p*77, 768)
             
             # Latent input must also be doubled
             x_in = torch.cat([x_t_latent, x_t_latent], dim=0) # (2*T, 4, h, w)
             t_list = torch.cat([self.sub_timesteps_tensor, self.sub_timesteps_tensor], dim=0)
             
        else:
             cond_flat = self.prompt_embeds
             text_embeds = rearrange(cond_flat, '(t p) l d -> t (p l) d', p=p, t=T_batch)
             x_in = x_t_latent
             t_list = self.sub_timesteps_tensor


        # 2. Construct RCCA Mask Pyramid (JIT)
        # We need masks for resolutions 64x64, 32x32, 16x16, 8x8 (relative to latent).
        # Latent h, w are e.g. 64.
        # Attention maps will be H*W.
        # self.masks shape: (p, T, 1, h, w).
        # We need to map this to Attention Mask (BatchSize, QueryLen, KeyLen)
        # BatchSize = 2*T (if CFG).
        # QueryLen = h*w (spatial).
        # KeyLen = p*77.
        
        rcca_masks = {}
        resolutions = [1, 2, 4, 8] # Downsample factors: 64, 32, 16, 8.
        # Note: U-Net CrossAttn is at 64(1x), 32(2x), 16(4x). 8(8x) is Mid.
        
        # Prepare base masks: rearrange to (T, p, 1, h, w)
        base_masks = rearrange(self.masks, 'p t c h w -> t p c h w')
        if self.guidance_scale > 1.0 and self.cfg_type in ('initialize', 'full'):
             # Duplicate for Uncond (apply same mask to uncond? Or full mask?)
             # Usually we want Negative Prompt to also be localized.
             base_masks = torch.cat([base_masks, base_masks], dim=0) # (2*T, p, 1, h, w)

        h_latent = x_t_latent.shape[-2]
        w_latent = x_t_latent.shape[-1]

        for scale_idx, scale in enumerate(resolutions):
             h_curr = h_latent // scale
             w_curr = w_latent // scale
             num_pixels = h_curr * w_curr
             
             # Downsample masks
             # Input: (TotalBatch, p, 1, h, w) -> reshape to (TotalBatch*p, 1, h, w) for interpolate
             bs_masks = base_masks.shape[0]
             flat_masks = rearrange(base_masks, 'b p c h w -> (b p) c h w')
             
             masks_res = F.interpolate(flat_masks, size=(h_curr, w_curr), mode='nearest')
             
             # Reshape back and Flatten spatial
             # (TotalBatch, p, 1, h_small, w_small)
             masks_res = rearrange(masks_res, '(b p) c h w -> b p (c h w)', b=bs_masks, p=p)
             
             # Now construct the Attention Bias Matrix (B, Q, K) = (B, H*W, P*77)
             # Start with -inf (masked)
             # Where mask is 1, set to 0.
             
             # Note: self.masks are 0 or 1 (or soft). 
             # For RCCA binary: 1=Attend, 0=Block.
             # Log(M): 1->0, 0->-inf.
             # Clamp to avoid log(0)
             
             # Optimize: We want (B, Pixel, P*77).
             # We have (B, P, Pixel).
             # Transpose to (B, Pixel, P).
             masks_res = masks_res.transpose(1, 2) # (B, Pixel, P)
             
             # Repeat for 77 tokens
             masks_res = masks_res.unsqueeze(-1).repeat(1, 1, 1, 77) # (B, Pixel, P, 77)
             masks_res = rearrange(masks_res, 'b x p l -> b x (p l)') # (B, Pixel, P*77)
             
             # Convert to log-space bias
             # Use a small epsilon for 0 values to get large negative number
             # masks_res values are in [0, 1]
             # bias = ln(mask + eps). 
             # If mask is 1, ln(1)=0. If mask is 0, ln(eps) ~ -inf.
             
             bias = torch.log(masks_res + 1e-5) # 1e-5 gives -11.5. Softmax will handle it? 
             # Standard masking uses -1e4 or similar.
             # Let's clean it up:
             bias = torch.where(masks_res > 0.5, torch.zeros_like(masks_res), torch.tensor(-10000.0, dtype=masks_res.dtype, device=masks_res.device))
             
             rcca_masks[num_pixels] = bias


        # 3. Forward Pass
        model_pred = self.unet(
            x_in,
            t_list,
            encoder_hidden_states=text_embeds,
            return_dict=False,
            cross_attention_kwargs={'rcca_masks': rcca_masks}
        )[0]
        
        
        # 4. Handle Output
        if self.guidance_scale > 1.0 and self.cfg_type in ('initialize', 'full'):
             noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
             
             # Update stock noise (legacy support, might act weird but keeping for safety)
             # Originally kept for 'self'/'initialize' CFG types.
             # If P changed, stock_noise_ shape logic in original code was complex.
             # Here we simplified to Single Pass.
             # We need to maintain self.stock_noise_ shape for next steps if needed?
             # self.stock_noise is (T, 4, h, w).
             # self.stock_noise_ was (T*p, ...).
             # We don't use stock_noise_ anymore in this simplified flow except for the 'self' correction.
             # If we want to support 'self' correction, we need to adapt it. 
             # For now, standard CFG:
             model_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
             model_pred = model_pred
             
             
        # 5. Denoise Step
        # compute the previous noisy sample x_t -> x_t-1
        # No averaging needed! Each pixel is already "pure".
        denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)


        # 6. Noise Addition for Stream Batch (Shift happens outside this func usually? No, SemanticDraw pipeline handles shifting?)
        # SemanticDraw.pdf says: "Slot 1 emits finished... Slot 2 moves to Slot 1".
        # The pipeline loop handles the shifting of slots (x_t_latent_buffer).
        # We just return the new latent.
        
        # Background handling (Legacy "Bootstrapping" replacement per marco.md)
        # "RCCA treats the background as the inverse of the foreground masks... logic handled in mask generation."
        # We handled it by including Background Prompt in the P prompts and generating a Background Mask.
        # So we don't need explicit background mixing here.
        
        # However, we must return `latent`.
        
        return denoised_batch

    @torch.no_grad()
    def __call__(
        self,
        no_decode: bool = False,
        ignore_check_ready: bool = False,
    ) -> Optional[Union[torch.Tensor, Image.Image]]:
        if not ignore_check_ready and not self.check_ready():
            return
        if not ignore_check_ready and self.is_dirty:
            print("I'm dirty!")
            self.commit()
            self.flush()

        if self.seedfix:
            self.reset_seed(self.generator, self.seed)
            latent = self.init_noise[:1]
        else:
            latent = torch.randn((1, self.unet.config.in_channels, self.latent_height, self.latent_width),
                dtype=self.dtype, device=self.device)  # (1, 4, h, w)
        latent = torch.cat((latent, self.x_t_latent_buffer), dim=0)  # (t, 4, h, w)
        self.stock_noise = torch.cat((self.init_noise[:1], self.stock_noise[:-1]), dim=0)  # (t, 4, h, w)
        if self.cfg_type in ('self', 'initialize'):
            self.stock_noise_ = self.stock_noise.repeat_interleave(self.num_layers, dim=0)  # (T * p, 77, 768)

        x_0_pred_batch = self.unet_step(latent)

        latent = x_0_pred_batch[-1:]
        self.x_t_latent_buffer = (
            self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
            + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
        )

        # For pipeline flushing.
        if no_decode:
            return latent

        imgs = self.decode_latents(latent.half())  # (1, 3, H, W)
        img = T.ToPILImage()(imgs[0].cpu())
        return img

    def flush(self) -> None:
        for _ in self.t_list:
            self(True, True)
        self.ready_checklist['flushed'] = True
