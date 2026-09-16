from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_unflatten

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.bonsai_image.weights.bonsai_image_weight_definition import BonsaiImageWeightDefinition
from mflux.models.common.config import ModelConfig
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_transformer.klein_fast import (
    Flux2KleinFastTransformer,
    Flux2KleinMegakernelSpec,
    find_packed_artifact_dir,
    load_klein_fast_packed_weights_from_disk,
)
from mflux.models.flux2.model.flux2_transformer.klein_fast.blocks import _require_native_quantized_matmul
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE


class BonsaiImageInitializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        del lora_paths, lora_scales
        if quantize is not None:
            raise ValueError(
                "Bonsai checkpoints are already packed low-bit MLX artifacts. "
                "Omit --quantize/-q when generating from Bonsai."
            )

        path = model_path if model_path else model_config.model_name
        root_path = PathResolution.resolve(
            path=path,
            patterns=BonsaiImageWeightDefinition.get_download_patterns(),
        )
        if root_path is None:
            raise FileNotFoundError(f"Unable to resolve Bonsai model path: {path}")

        BonsaiImageInitializer._init_config(model, model_config)
        model.model_path = path
        BonsaiImageInitializer._init_tokenizers(model, path)
        BonsaiImageInitializer._init_models(model, root_path)
        BonsaiImageInitializer._load_packed_transformer(model, root_path)
        BonsaiImageInitializer._apply_vae(model, path)
        BonsaiImageInitializer._load_text_encoder(model, root_path)
        model.lora_paths = None
        model.lora_scales = None

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=BonsaiImageWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model, root_path: Path) -> None:
        model.vae = Flux2VAE()
        model.text_encoder = BonsaiImageInitializer._new_text_encoder(root_path)
        model.transformer = None

    @staticmethod
    def _apply_vae(model, model_path: str) -> None:
        weights = WeightLoader.load(
            weight_definition=BonsaiImageWeightDefinition,
            model_path=model_path,
        )
        WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=None,
            weight_definition=BonsaiImageWeightDefinition,
            models={"vae": model.vae},
        )

    @staticmethod
    def _new_text_encoder(root_path: Path) -> Qwen3TextEncoder:
        config_path = root_path / "text_encoder-mlx-4bit" / "config.json"
        config = json.loads(config_path.read_text())
        return Qwen3TextEncoder(
            vocab_size=int(config.get("vocab_size", 151936)),
            hidden_size=int(config.get("hidden_size", 2560)),
            num_hidden_layers=int(config.get("num_hidden_layers", 36)),
            num_attention_heads=int(config.get("num_attention_heads", 32)),
            num_key_value_heads=int(config.get("num_key_value_heads", 8)),
            intermediate_size=int(config.get("intermediate_size", 9728)),
            max_position_embeddings=int(config.get("max_position_embeddings", 40960)),
            rope_theta=float(config.get("rope_theta", 1000000.0)),
            rms_norm_eps=float(config.get("rms_norm_eps", 1e-6)),
            head_dim=int(config.get("head_dim", 128)),
            attention_bias=bool(config.get("attention_bias", False)),
        )

    @staticmethod
    def _load_text_encoder(model, root_path: Path) -> None:
        text_encoder_dir = root_path / "text_encoder-mlx-4bit"
        config = json.loads((text_encoder_dir / "config.json").read_text())
        quant_config = config.get("quantization_config") or config.get("quantization") or {}
        bits = int(quant_config.get("bits", 4))
        group_size = int(quant_config.get("group_size", 64))
        if bits != 4:
            raise ValueError(f"Bonsai text_encoder-mlx-4bit must use 4-bit weights, got bits={bits}")

        nn.quantize(
            model.text_encoder,
            class_predicate=lambda _path, module: hasattr(module, "to_quantized"),
            bits=bits,
            group_size=group_size,
        )

        raw = BonsaiImageInitializer._load_safetensors_dir(text_encoder_dir)
        stripped = {}
        for key, value in raw.items():
            if key.startswith("model."):
                key = key[len("model.") :]
            stripped[key] = value
        model.text_encoder.update(tree_unflatten(list(stripped.items())), strict=False)

    @staticmethod
    def _load_packed_transformer(model, root_path: Path) -> None:
        packed_dir = find_packed_artifact_dir(root_path)
        if packed_dir is None:
            raise FileNotFoundError(f"Bonsai model is missing transformer-packed-mflux/: {root_path}")

        quant_config = json.loads((packed_dir / "quantization_config.json").read_text())
        bits = int(quant_config["bits"])
        group_size = int(quant_config["group_size"])
        BonsaiImageInitializer._ensure_packed_affine_runtime(bits=bits, group_size=group_size)

        config = json.loads((packed_dir / "config.json").read_text())
        spec = Flux2KleinMegakernelSpec(
            num_double_blocks=int(config.get("num_layers", 5)),
            num_single_blocks=int(config.get("num_single_layers", 20)),
            dim=int(config.get("num_attention_heads", 24)) * int(config.get("attention_head_dim", 128)),
            num_heads=int(config.get("num_attention_heads", 24)),
            head_dim=int(config.get("attention_head_dim", 128)),
            mlp_ratio=float(config.get("mlp_ratio", 3.0)),
            layer_norm_eps=float(config.get("eps", 1e-6)),
            rms_norm_eps=float(config.get("eps", 1e-6)),
            rope_theta=int(config.get("rope_theta", 2000)),
            axes_dims_rope=tuple(config.get("axes_dims_rope", (32, 32, 32, 32))),
            in_channels=int(config.get("in_channels", 128)),
            context_dim=int(config.get("joint_attention_dim", 7680)),
        )
        weights = load_klein_fast_packed_weights_from_disk(packed_dir, spec, dtype=ModelConfig.precision)
        precision = "1bit" if bits == 1 else "2bit"
        transformer = Flux2KleinFastTransformer(
            weights=weights,
            precision=precision,
            group_size=group_size,
            patch_size=int(config.get("patch_size", 1)),
            in_channels=spec.in_channels,
            out_channels=int(config.get("out_channels") or spec.in_channels),
            num_layers=spec.num_double_blocks,
            num_single_layers=spec.num_single_blocks,
            attention_head_dim=spec.head_dim,
            num_attention_heads=spec.num_heads,
            joint_attention_dim=spec.context_dim,
            timestep_guidance_channels=int(config.get("pooled_projection_dim", 256)),
            mlp_ratio=spec.mlp_ratio,
            axes_dims_rope=spec.axes_dims_rope,
            rope_theta=spec.rope_theta,
            guidance_embeds=bool(config.get("guidance_embeds", False)),
            layer_norm_eps=spec.layer_norm_eps,
            rms_norm_eps=spec.rms_norm_eps,
        )

        raw = BonsaiImageInitializer._load_safetensors_dir(packed_dir)
        transformer.time_guidance_embed.linear_1.weight = raw[
            "time_guidance_embed.timestep_embedder.linear_1.weight"
        ].astype(ModelConfig.precision)
        transformer.time_guidance_embed.linear_2.weight = raw[
            "time_guidance_embed.timestep_embedder.linear_2.weight"
        ].astype(ModelConfig.precision)
        model.transformer = transformer
        model.bits = bits

    @staticmethod
    def _ensure_packed_affine_runtime(bits: int, group_size: int) -> None:
        try:
            _require_native_quantized_matmul(bits, group_size)
        except Exception as exc:  # noqa: BLE001
            if bits == 1:
                raise RuntimeError(
                    "Bonsai binary 1-bit checkpoints require a native MLX packed affine kernel for "
                    f"bits=1, group_size={group_size}. The active MLX runtime cannot execute it yet. "
                    "Use prism-ml/bonsai-image-ternary-4B-mlx-2bit, or retry after installing an MLX "
                    "runtime that supports 1-bit quantized_matmul."
                ) from exc
            raise RuntimeError(
                f"The active MLX runtime cannot execute packed affine matmul for bits={bits}, "
                f"group_size={group_size}."
            ) from exc

    @staticmethod
    def _load_safetensors_dir(path: Path) -> dict[str, mx.array]:
        raw: dict[str, mx.array] = {}
        shards = sorted(p for p in path.glob("*.safetensors") if not p.name.startswith("._"))
        if not shards:
            raise FileNotFoundError(f"No safetensors files found in {path}")
        for shard in shards:
            raw.update(mx.load(str(shard)))
        return raw
