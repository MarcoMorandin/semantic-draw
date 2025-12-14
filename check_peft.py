
import sys
import torch
print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")

try:
    import peft
    print(f"PEFT imported successfully. Version: {peft.__version__}")
except ImportError as e:
    print(f"Failed to import peft: {e}")

try:
    import diffusers
    print(f"Diffusers imported successfully. Version: {diffusers.__version__}")
    from diffusers.utils import is_peft_available, USE_PEFT_BACKEND
    print(f"diffusers.utils.is_peft_available(): {is_peft_available()}")
    print(f"diffusers.utils.USE_PEFT_BACKEND: {USE_PEFT_BACKEND}")
except ImportError as e:
    print(f"Failed to import/check diffusers: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
