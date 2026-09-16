"""Load published `AutoencoderTiny` checkpoints into the MLX port.

Checkpoint keys follow the diffusers layout (`decoder.layers.3.conv.0.weight`).
MLX's `nn.Sequential` nests its children under `.layers`, and MLX convolutions are
channels-last, so mapping is a key rewrite plus an `OIHW -> OHWI` weight transpose.
"""

import json
import re
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.download_policy import downloads_enabled, raise_download_required
from mflux.models.common.preview.tiny_autoencoder import TinyAutoencoder

_WEIGHT_PATTERNS = ["*.safetensors", "config.json"]
_SEQUENTIAL_KEY = re.compile(r"^(encoder|decoder)\.layers\.(\d+)\.(.*)$")
_BLOCK_SUBMODULE_KEY = re.compile(r"^(conv|pool)\.(\d+)\.(weight|bias)$")


class TinyAutoencoderLoader:
    @staticmethod
    def load(
        repo_id: str,
        *,
        with_encoder: bool = False,
        config_overrides: dict | None = None,
        weight_file: str | None = None,
    ) -> TinyAutoencoder:
        """Load a published checkpoint. `config_overrides` supplies the architecture for
        repositories that ship weights without a diffusers config (for example taef2), and
        `weight_file` pins the tensor file when a repository publishes more than one."""
        root_path = TinyAutoencoderLoader._resolve_root(repo_id)
        config = TinyAutoencoderLoader._read_config(root_path, overrides=config_overrides)
        model = TinyAutoencoder(**config, with_encoder=with_encoder)
        weights = TinyAutoencoderLoader._read_weights(root_path, weight_file=weight_file)
        mapped = TinyAutoencoderLoader._map_weights(weights, with_encoder=with_encoder)
        TinyAutoencoderLoader._assert_coverage(model, mapped, repo_id=repo_id)
        model.update(tree_unflatten(list(mapped.items())))
        mx.eval(model.parameters())
        return model

    @staticmethod
    def _resolve_root(repo_id: str) -> Path:
        local_path = Path(repo_id).expanduser()
        if local_path.is_dir():
            return local_path

        # Deferred import: huggingface_hub is only needed once weights actually resolve (0088).
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import LocalEntryNotFoundError

        try:
            return Path(snapshot_download(repo_id=repo_id, allow_patterns=_WEIGHT_PATTERNS, local_files_only=True))
        except LocalEntryNotFoundError:
            if not downloads_enabled():
                raise_download_required(repo_id, artifact="preview decoder")
            return Path(snapshot_download(repo_id=repo_id, allow_patterns=_WEIGHT_PATTERNS))

    @staticmethod
    def _read_config(root_path: Path, *, overrides: dict | None = None) -> dict:
        raw = TinyAutoencoderLoader._read_config_file(root_path)
        if raw is None:
            if not overrides:
                raise FileNotFoundError(
                    f"No usable tiny autoencoder config in {root_path}; this checkpoint needs an explicit architecture."
                )
            raw = {}
        raw = {**raw, **(overrides or {})}
        if raw.get("act_fn", "relu") != "relu":
            raise ValueError(f"Unsupported tiny autoencoder activation: {raw.get('act_fn')!r}")
        return {
            "use_midblock_gn": bool(raw.get("use_midblock_gn", False)),
            "in_channels": int(raw.get("in_channels", 3)),
            "out_channels": int(raw.get("out_channels", 3)),
            "latent_channels": int(raw["latent_channels"]),
            "encoder_block_out_channels": tuple(raw.get("encoder_block_out_channels", (64, 64, 64, 64))),
            "decoder_block_out_channels": tuple(raw.get("decoder_block_out_channels", (64, 64, 64, 64))),
            "num_encoder_blocks": tuple(raw.get("num_encoder_blocks", (1, 3, 3, 3))),
            "num_decoder_blocks": tuple(raw.get("num_decoder_blocks", (3, 3, 3, 1))),
            "upsampling_scaling_factor": int(raw.get("upsampling_scaling_factor", 2)),
            # Published configs carry these as null or omit them; a real 0.0 must survive.
            "scaling_factor": float(raw.get("scaling_factor") if raw.get("scaling_factor") is not None else 1.0),
            "shift_factor": float(raw.get("shift_factor") if raw.get("shift_factor") is not None else 0.0),
        }

    @staticmethod
    def _read_config_file(root_path: Path) -> dict | None:
        config_path = root_path / "config.json"
        if not config_path.exists():
            return None
        try:
            parsed = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            # Some repositories publish only weights; a non-JSON config.json is not a config.
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _read_weights(root_path: Path, *, weight_file: str | None = None) -> dict[str, mx.array]:
        files = [root_path / weight_file] if weight_file else sorted(root_path.glob("*.safetensors"))
        if not files or not all(path.exists() for path in files):
            raise FileNotFoundError(f"No safetensors weights found in {root_path}")
        weights: dict[str, mx.array] = {}
        for file_path in files:
            weights.update(mx.load(str(file_path)))
        return weights

    @staticmethod
    def _map_weights(weights: dict[str, mx.array], *, with_encoder: bool) -> dict[str, mx.array]:
        mapped: dict[str, mx.array] = {}
        for key, value in weights.items():
            match = _SEQUENTIAL_KEY.match(key)
            if match is None:
                continue
            component, index, remainder = match.group(1), match.group(2), match.group(3)
            if component == "encoder" and not with_encoder:
                continue
            submodule = _BLOCK_SUBMODULE_KEY.match(remainder)
            if submodule is not None:
                remainder = f"{submodule.group(1)}.layers.{submodule.group(2)}.{submodule.group(3)}"
            mapped[f"{component}.layers.layers.{index}.{remainder}"] = TinyAutoencoderLoader._to_mlx_layout(value)
        return mapped

    @staticmethod
    def _to_mlx_layout(value: mx.array) -> mx.array:
        # Convolution kernels: torch (out, in, kH, kW) -> MLX (out, kH, kW, in).
        return mx.transpose(value, (0, 2, 3, 1)) if value.ndim == 4 else value

    @staticmethod
    def _assert_coverage(model: TinyAutoencoder, mapped: dict[str, mx.array], *, repo_id: str) -> None:
        expected = {key: value.shape for key, value in tree_flatten(model.parameters())}
        missing = sorted(set(expected) - set(mapped))
        extra = sorted(set(mapped) - set(expected))
        mismatched = [key for key in set(expected) & set(mapped) if tuple(mapped[key].shape) != tuple(expected[key])]
        if missing or extra or mismatched:
            raise ValueError(
                f"Tiny autoencoder weight mismatch for {repo_id}: "
                f"missing={len(missing)} ({', '.join(missing[:3]) or 'none'}); "
                f"extra={len(extra)} ({', '.join(extra[:3]) or 'none'}); "
                f"shape-mismatch={len(mismatched)} ({', '.join(mismatched[:3]) or 'none'})"
            )
