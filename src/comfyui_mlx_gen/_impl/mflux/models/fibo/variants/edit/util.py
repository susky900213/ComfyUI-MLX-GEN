import gc
import json
from pathlib import Path

import mlx.core as mx
from PIL import Image

from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.fibo.latent_creator.fibo_latent_creator import FiboLatentCreator
from mflux.models.fibo.model.fibo_vae.wan_2_2_vae import Wan2_2_VAE
from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM
from mflux.utils.image_util import ImageUtil
from mflux.utils.prompt_util import PromptUtil
from mflux.utils.scale_factor import ScaleFactor

FIBO_EDIT_RMBG_DEFAULT_EDIT_INSTRUCTION = (
    "Generate a detailed grayscale alpha matte. Map the opaque foreground to white "
    "and the background to black. Produce soft, anti-aliased grayscale gradients at the "
    "edges of the subject to represent fine details and transparency."
)
FIBO_EDIT_RMBG_DEFAULT_JSON_PROMPT = json.dumps({"edit_instruction": FIBO_EDIT_RMBG_DEFAULT_EDIT_INSTRUCTION})
FIBO_EDIT_PROMPTIFIER_MODEL_ID = "briaai/FIBO-edit-prompt-to-JSON"
FIBO_EDIT_DIMENSION_MULTIPLE = 16
FIBO_EDIT_PREFERRED_RESOLUTIONS_1024: tuple[tuple[int, int], ...] = (
    (832, 1248),
    (880, 1184),
    (912, 1136),
    (1024, 1024),
    (1136, 912),
    (1184, 880),
    (1216, 848),
    (1248, 832),
    (1248, 832),
    (1264, 816),
    (1296, 800),
    (1360, 768),
)


class FiboEditUtil:
    @staticmethod
    def get_json_prompt_for_edit(
        args,
        quantize: int | None,
        default_json_prompt_if_missing: str | None = None,
    ) -> str:
        prompt = PromptUtil.read_prompt(args)
        missing = prompt is None or (isinstance(prompt, str) and not prompt.strip())
        if missing:
            if default_json_prompt_if_missing is not None:
                return FiboEditUtil.ensure_edit_instruction(default_json_prompt_if_missing)
            raise ValueError(
                "FIBO edit requires an edit instruction via --prompt/--prompt-file, or a JSON prompt from metadata."
            )

        try:
            return FiboEditUtil.ensure_edit_instruction(prompt)
        except ValueError:
            try:
                json.loads(prompt)
            except (TypeError, json.JSONDecodeError):
                pass
            else:
                raise

        if getattr(args, "image_path", None) is None:
            raise ValueError("Edit mode requires --image-path when prompt is not valid JSON.")
        if getattr(args, "mask_path", None) is not None:
            raise ValueError(
                "Masked FIBO edit requires a JSON prompt with `edit_instruction`; "
                "local masked prompt-to-JSON conversion is not supported."
            )

        print("Preparing FIBO edit JSON prompt with the local VLM before denoising...", flush=True)
        image = ImageUtil.load_image(args.image_path)

        vlm = FiboVLM(model_id=FIBO_EDIT_PROMPTIFIER_MODEL_ID, quantize=quantize)
        try:
            return vlm.edit(
                image=image,
                edit_instruction=prompt,
                use_mask=getattr(args, "mask_path", None) is not None,
                seed=42,
            )
        finally:
            del vlm
            gc.collect()
            mx.clear_cache()

    @staticmethod
    def parse_json_prompt(prompt: str | dict) -> dict:
        if isinstance(prompt, dict):
            return dict(prompt)
        try:
            value = json.loads(prompt)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("FIBO edit prompt must be a valid JSON string.") from exc

        if not isinstance(value, dict):
            raise ValueError("FIBO edit prompt JSON must be an object.")
        return value

    @staticmethod
    def ensure_edit_instruction(prompt: str | dict, edit_instruction: str | None = None) -> str:
        prompt_dict = FiboEditUtil.parse_json_prompt(prompt)
        if "edit_instruction" in prompt_dict and prompt_dict["edit_instruction"]:
            return json.dumps(prompt_dict)

        if edit_instruction is None or not edit_instruction.strip():
            raise ValueError("FIBO edit prompt JSON must include `edit_instruction`.")

        prompt_dict["edit_instruction"] = edit_instruction.strip()
        return json.dumps(prompt_dict)

    @staticmethod
    def load_edit_image(
        image_path: Path | str,
        width: int,
        height: int,
        mask_path: Path | str | None = None,
    ) -> Image.Image:
        image = ImageUtil.load_image(image_path)
        if mask_path is None:
            return ImageUtil.scale_to_dimensions(image, width, height)

        mask_image = Image.open(mask_path).convert("L")
        if mask_image.size != image.size:
            raise ValueError("Mask and image must have the same size.")

        masked_image = FiboEditUtil._composite_mask_on_image(mask=mask_image, image=image)
        return ImageUtil.scale_to_dimensions(masked_image, width, height)

    @staticmethod
    def resolve_preferred_canvas_size(
        image_path: Path | str,
        width: int | ScaleFactor | None,
        height: int | ScaleFactor | None,
    ) -> tuple[int | ScaleFactor | None, int | ScaleFactor | None]:
        if width is not None or height is not None:
            return width, height

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        source_ratio = image_width / image_height
        return min(
            FIBO_EDIT_PREFERRED_RESOLUTIONS_1024,
            key=lambda size: abs(size[0] / size[1] - source_ratio),
        )

    @staticmethod
    def encode_conditioning_image(
        vae: Wan2_2_VAE,
        image: Image.Image,
        height: int,
        width: int,
        tiling_config=None,
        dtype: mx.Dtype | None = None,
    ) -> mx.array:
        image_array = ImageUtil.to_array(image=image)
        if dtype is not None:
            image_array = image_array.astype(dtype)
        image_latents = VAEUtil.encode(vae=vae, image=image_array, tiling_config=tiling_config)
        if dtype is not None:
            image_latents = image_latents.astype(dtype)
        return FiboLatentCreator.pack_latents(latents=image_latents, height=height, width=width)

    @staticmethod
    def create_conditioning_image_ids(height: int, width: int, dtype: mx.Dtype) -> mx.array:
        latent_height = height // 16
        latent_width = width // 16
        row_indices = mx.arange(0, latent_height, dtype=dtype)[:, None]
        row_indices = mx.broadcast_to(row_indices, (latent_height, latent_width))
        col_indices = mx.arange(0, latent_width, dtype=dtype)[None, :]
        col_indices = mx.broadcast_to(col_indices, (latent_height, latent_width))
        ones_channel = mx.ones((latent_height, latent_width), dtype=dtype)
        latent_image_ids = mx.stack([ones_channel, row_indices, col_indices], axis=-1)
        latent_image_ids = mx.reshape(latent_image_ids, (1, latent_height * latent_width, 3))
        return latent_image_ids

    @staticmethod
    def _composite_mask_on_image(mask: Image.Image, image: Image.Image) -> Image.Image:
        gray_img = Image.new("RGB", image.size, (128, 128, 128))
        return Image.composite(gray_img, image.convert("RGB"), mask.convert("L"))

    @staticmethod
    def build_rgba_composite_image(source_image_path: Path | str, matte_image: Image.Image) -> Image.Image:
        base = Image.open(source_image_path).convert("RGB").copy()
        alpha = matte_image.convert("L").resize(base.size, Image.LANCZOS)
        base.putalpha(alpha)
        return base
