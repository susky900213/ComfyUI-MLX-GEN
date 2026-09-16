import os
from pathlib import Path

import platformdirs

from mflux.models.common.config.inference_defaults import MODEL_INFERENCE_STEPS as MODEL_INFERENCE_STEPS

BATTERY_PERCENTAGE_STOP_LIMIT = 5
CONTROLNET_STRENGTH = 0.4
DEFAULT_DEV_FILL_GUIDANCE = 30
DEFAULT_DEPTH_GUIDANCE = 10
DIMENSION_STEP_PIXELS = 16
GUIDANCE_SCALE = 3.5
GUIDANCE_SCALE_KONTEXT = 2.5
HEIGHT, WIDTH = 1024, 1024
IMAGE_STRENGTH = 0.4
MODEL_CHOICES = [
    "dev",
    "schnell",
    "krea-dev",
    "dev-krea",
    "qwen",
    "qwen-image",
    "qwen-image-edit",
    "qwen-image-edit-2509",
    "qwen-image-edit-2511",
    "qwen-edit",
    "qwen-edit-plus",
    "qwen-edit-2509",
    "qwen-edit-2511",
    "fibo",
    "fibo-lite",
    "fibo-edit",
    "fibo-edit-rmbg",
    "z-image",
    "z-image-turbo",
    "ernie-image-turbo",
    "seedvr2",
    "seedvr2-3b",
    "seedvr2-7b",
    "seedvr2-7b-sharp",
    "swiftvr",
    "wan2.2-ti2v-5b",
    "bonsai-image-ternary",
    "bonsai-image-binary",
    "flux2-klein-4b",
    "flux2-klein-9b",
    "flux2-klein-base-4b",
    "flux2-klein-base-9b",
]
QUANTIZE_CHOICES = [3, 5, 4, 6, 8]

if os.environ.get("MFLUX_CACHE_DIR"):
    MFLUX_CACHE_DIR = Path(os.environ["MFLUX_CACHE_DIR"]).resolve()
else:
    MFLUX_CACHE_DIR = Path(platformdirs.user_cache_dir(appname="mflux"))

MFLUX_LORA_CACHE_DIR = MFLUX_CACHE_DIR / "loras"
