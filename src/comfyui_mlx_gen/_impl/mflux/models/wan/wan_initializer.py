import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.lora_compatibility import LoRACompatibility
from mflux.models.common.lora.mapping.lora_loader import LoRAApplicationError, LoRAApplicationResult, LoRALoader
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.wan.model.wan_transformer import WanTransformer
from mflux.models.wan.model.wan_vae import Wan2_2_VAE
from mflux.models.wan.weights import WanWeightDefinition
from mflux.models.wan.weights.wan_lora_mapping import WanLoRAMapping


@dataclass(frozen=True)
class _WanComponentSources:
    root_path: Path
    component_roots: dict[str, Path]
    provenance: dict[str, dict[str, str]]
    factored: bool


class WanInitializer:
    # The SVI error-recycling LoRAs retrain the conditioning convention; the
    # reference loads them at alpha=1 on top of the base experts, and our
    # LoRALinear applies scale x (B @ A) with no rank divisor when the file
    # carries no alpha keys - so 1.0 reproduces the reference exactly.
    SVI_LORA_SCALE = 1.0

    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        lora_target_roles: list[str] | None = None,
        svi_lora_high_path: str | None = None,
        svi_lora_low_path: str | None = None,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        LoRACompatibility.validate_for_model_config(
            model_config=model_config,
            selected_model=path,
            lora_paths=lora_paths,
        )
        weight_definition = WanWeightDefinition.for_config(model_config)
        sources = WanInitializer._resolve_component_sources(
            model_config=model_config,
            model_path=model_path,
            weight_definition=weight_definition,
        )
        if sources.factored:
            WanInitializer._validate_factored_source_config(
                base_root=sources.root_path,
                transformer_root=sources.component_roots["transformer"],
                model_config=model_config,
            )
        else:
            WanInitializer._validate_source_config(sources.root_path, model_config)
        WanInitializer._init_config(
            model,
            model_config,
            sources.root_path,
            weight_definition,
            component_roots=sources.component_roots,
            component_source_provenance=sources.provenance,
            factored_component_sources=sources.factored,
        )
        # Reload spec (0089 e4): keep the ORIGINAL quantize request so a released
        # high-noise expert can be rebuilt exactly as at init (model.bits only
        # records the resolved level, which loses the stored-vs-requested source).
        model.quantize_arg = quantize
        tokenizer_root = sources.component_roots.get("tokenizer", sources.root_path)
        WanInitializer._init_tokenizers(model, str(tokenizer_root), weight_definition)
        WanInitializer._init_models(model, model_config)
        WanInitializer._load_and_apply_weights(model, sources.root_path, quantize, weight_definition)
        WanInitializer._apply_lora(
            model,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            lora_target_roles=lora_target_roles,
        )
        WanInitializer._apply_svi_loras(
            model,
            svi_lora_high_path=svi_lora_high_path,
            svi_lora_low_path=svi_lora_low_path,
        )

    @staticmethod
    def reload_high_noise_transformer(model) -> None:
        # Per-item A14B reload (0089 e4): rebuild ONLY the high-noise expert from
        # the same checkpoint and quantize request captured at init, then re-fuse
        # its role's LoRAs on the fresh module in the original order. The q8
        # normalization notice below may print once per reload (= once per batch
        # item) - accepted as truthful load output.
        weight_definition = model.weight_definition
        component = next(
            (candidate for candidate in weight_definition.get_components() if candidate.name == "transformer"),
            None,
        )
        if component is None:
            raise ValueError("Wan weight definition has no 'transformer' component to reload.")
        print("Reloading Wan high-noise transformer for the next high-noise phase...")
        transformer = WanTransformer(**WanInitializer._transformer_kwargs(model.model_config))
        component_root = getattr(model, "component_roots", {}).get(component.name, model.root_path)
        component_weights, q_level, version = WeightLoader._load_component(
            root_path=component_root,
            component=component,
        )
        WanInitializer._normalize_runtime_sensitive_q8_paths(
            component_name=component.name,
            component_weights=component_weights,
            q_level=q_level,
        )
        WanInitializer._validate_component_quantization_layout(component.name, component_weights, q_level)
        loaded_weights = LoadedWeights(
            components={component.name: component_weights},
            meta_data=MetaData(quantization_level=q_level, mflux_version=version),
        )
        bits = WeightApplier.apply_and_quantize_single(
            weights=loaded_weights,
            model=transformer,
            component=component,
            quantize_arg=getattr(model, "quantize_arg", None),
            quantization_predicate=weight_definition.quantization_predicate,
        )
        if bits != model.bits:
            raise ValueError(
                f"Wan high-noise reload resolved quantization {bits}, but the model was loaded at {model.bits}. "
                "The checkpoint appears to have changed on disk; construct a fresh Wan model instance."
            )
        del loaded_weights
        del component_weights
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        WanInitializer._reapply_high_noise_loras(model, transformer)
        model.transformer = transformer

    @staticmethod
    def _init_config(
        model,
        model_config: ModelConfig,
        root_path: Path,
        weight_definition: WanWeightDefinition,
        component_roots: dict[str, Path] | None = None,
        component_source_provenance: dict[str, dict[str, str]] | None = None,
        factored_component_sources: bool = False,
    ) -> None:
        model.model_config = model_config
        model.root_path = root_path
        model.component_roots = dict(component_roots or {})
        model.component_source_provenance = dict(component_source_provenance or {})
        model.factored_component_sources = factored_component_sources
        model.weight_definition = weight_definition
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model.prompt_embed_cache = {}
        model.image_condition_cache = {}

    @staticmethod
    def _load_weights(model_path: str, weight_definition: WanWeightDefinition) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=weight_definition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str, weight_definition: WanWeightDefinition) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=weight_definition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model, model_config: ModelConfig) -> None:
        transformer_kwargs = WanInitializer._transformer_kwargs(model_config)
        model.transformer = WanTransformer(**transformer_kwargs)
        model.transformer_2 = (
            WanTransformer(**transformer_kwargs)
            if model_config.transformer_overrides.get("has_transformer_2", False)
            else None
        )
        model.vae = Wan2_2_VAE(**WanInitializer._vae_kwargs(model_config))

    @staticmethod
    def _apply_weights(
        model,
        weights: LoadedWeights,
        quantize: int | None,
        weight_definition: WanWeightDefinition,
    ) -> None:
        models = {
            "transformer": model.transformer,
            "vae": model.vae,
        }
        if model.transformer_2 is not None:
            models["transformer_2"] = model.transformer_2
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=weight_definition,
            models=models,
        )

    @staticmethod
    def _load_and_apply_weights(
        model,
        root_path: Path,
        quantize: int | None,
        weight_definition: WanWeightDefinition,
    ) -> None:
        bits = None
        bits_resolved = False

        for component in weight_definition.get_components():
            component_model = getattr(model, component.model_attr or component.name, None)
            if component_model is None:
                continue

            component_root = getattr(model, "component_roots", {}).get(component.name, root_path)
            component_weights, q_level, version = WeightLoader._load_component(
                root_path=component_root,
                component=component,
            )
            if component.name == "transformer":
                # Reload spec (0089 e4): the auto-release default keys on whether
                # the high expert came from a disk-prequantized package (reload =
                # mmap read) versus runtime quantization (reload re-quantizes 14B).
                model.transformer_stored_q_level = q_level
            WanInitializer._normalize_runtime_sensitive_q8_paths(
                component_name=component.name,
                component_weights=component_weights,
                q_level=q_level,
            )
            WanInitializer._validate_component_quantization_layout(component.name, component_weights, q_level)
            loaded_weights = LoadedWeights(
                components={component.name: component_weights},
                meta_data=MetaData(quantization_level=q_level, mflux_version=version),
            )
            component_bits = WeightApplier.apply_and_quantize_single(
                weights=loaded_weights,
                model=component_model,
                component=component,
                quantize_arg=quantize,
                quantization_predicate=weight_definition.quantization_predicate,
            )
            mismatch_error = None
            if component.skip_quantization and component_bits is None:
                pass
            elif not bits_resolved:
                bits = component_bits
                bits_resolved = True
            elif component_bits != bits:
                mismatch_error = ValueError(
                    "Wan component quantization mismatch: "
                    f"{component.name} resolved to {component_bits}, but earlier components resolved to {bits}."
                )
            del loaded_weights
            del component_weights
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
            if mismatch_error is not None:
                raise mismatch_error

        model.bits = bits

    @staticmethod
    def _apply_lora(
        model,
        lora_paths: list[str] | None,
        lora_scales: list[float] | None,
        lora_target_roles: list[str] | None,
    ) -> None:
        resolved_roles = WanInitializer._resolve_lora_roles(
            model,
            lora_paths=lora_paths,
            lora_target_roles=lora_target_roles,
        )
        if not lora_paths:
            model.lora_application_result = LoRAApplicationResult(resolved_paths=[], resolved_scales=[], reports=())
            model.lora_application_reports = ()
            model.lora_paths = []
            model.lora_scales = []
            model.lora_target_roles = []
            return

        if lora_scales is not None and len(lora_scales) != len(lora_paths):
            raise LoRAApplicationError(
                f"Number of LoRA scales ({len(lora_scales)}) must match number of LoRA files ({len(lora_paths)})."
            )

        results: list[LoRAApplicationResult] = []
        for index, lora_path in enumerate(lora_paths):
            role = resolved_roles[index]
            transformer = WanInitializer._transformer_for_role(model, role)
            result = LoRALoader.load_and_apply_lora_detailed(
                lora_mapping=WanLoRAMapping.get_mapping(),
                transformer=transformer,
                lora_paths=[lora_path],
                lora_scales=None if lora_scales is None else [lora_scales[index]],
                role=role,
                state_dict_transform=WanInitializer._transform_wan_lora_state_dict,
            )
            results.append(result)

        model.lora_application_result = LoRAApplicationResult(
            resolved_paths=[path for result in results for path in result.resolved_paths],
            resolved_scales=[scale for result in results for scale in result.resolved_scales],
            reports=tuple(report for result in results for report in result.reports),
        )
        model.lora_application_reports = model.lora_application_result.reports
        model.lora_paths = model.lora_application_result.resolved_paths
        model.lora_scales = model.lora_application_result.resolved_scales
        model.lora_target_roles = resolved_roles

    @staticmethod
    def _apply_svi_loras(
        model,
        *,
        svi_lora_high_path: str | None,
        svi_lora_low_path: str | None,
    ) -> None:
        # SVI 2.0 Pro pack loading (0103). Kept SEPARATE from the generic LoRA
        # bookkeeping: metadata `lora_paths` must stay replayable through
        # --lora-paths, while the SVI pack replays through its own flags and
        # carries its own strict-match contract.
        if svi_lora_high_path is None and svi_lora_low_path is None:
            model.svi_lora_paths = []
            model.svi_lora_reports = ()
            return
        if svi_lora_high_path is None or svi_lora_low_path is None:
            raise LoRAApplicationError(
                "The SVI LoRA pair is indivisible: pass BOTH svi_lora_high_path and svi_lora_low_path "
                "(the error-recycling fine-tune targets the high- and low-noise experts together)."
            )
        if model.transformer_2 is None:
            raise LoRAApplicationError(
                f"{model.model_config.model_name} does not support the SVI LoRA pair: SVI 2.0 Pro targets "
                "the dual-expert Wan 2.2 A14B image-to-video model."
            )
        reports = []
        resolved_paths = []
        for lora_path, role in (
            (svi_lora_high_path, "high_noise_transformer"),
            (svi_lora_low_path, "low_noise_transformer"),
        ):
            result = LoRALoader.load_and_apply_lora_detailed(
                lora_mapping=WanLoRAMapping.get_mapping(),
                transformer=WanInitializer._transformer_for_role(model, role),
                lora_paths=[lora_path],
                lora_scales=[WanInitializer.SVI_LORA_SCALE],
                role=role,
                state_dict_transform=WanInitializer._transform_wan_lora_state_dict,
            )
            report = result.reports[0]
            WanInitializer._require_strict_svi_key_match(report)
            reports.append(report)
            resolved_paths.extend(result.resolved_paths)
        model.svi_lora_paths = resolved_paths
        model.svi_lora_reports = tuple(reports)

    @staticmethod
    def _require_strict_svi_key_match(report) -> None:
        # The generic loader warns-and-skips unmatched keys; for the SVI pack a
        # partial match means a partially-taught conditioning convention - the
        # exact silent-failure trap the redo doctrine gates against. Zero
        # tolerance, loud fail (doctrine B1: assert unmatched_key_count == 0).
        if report.unmatched_key_count != 0:
            raise LoRAApplicationError(
                f"SVI LoRA {report.resolved_path} ({report.role}) left {report.unmatched_key_count} of "
                f"{report.total_key_count} keys unmatched (matched {report.matched_key_count}). A partially "
                "applied SVI error-recycling LoRA silently corrupts the conditioning convention; refusing "
                "to continue. The file may not be an SVI Wan2.2-I2V-A14B pack, or its key format is "
                "unsupported."
            )

    @staticmethod
    def _reapply_high_noise_loras(model, transformer: WanTransformer) -> None:
        # Deterministic re-fusion (0089 e4): iterate the resolved paths/scales/roles
        # exactly as stored at init so the fused stack order (and any FusedLoRALinear
        # layering) reproduces the original weights bit for bit.
        lora_paths = getattr(model, "lora_paths", None) or []
        lora_scales = getattr(model, "lora_scales", None) or []
        lora_roles = getattr(model, "lora_target_roles", None) or []
        for lora_path, lora_scale, role in zip(lora_paths, lora_scales, lora_roles):
            if role not in ("transformer", "high_noise_transformer"):
                continue
            LoRALoader.load_and_apply_lora_detailed(
                lora_mapping=WanLoRAMapping.get_mapping(),
                transformer=transformer,
                lora_paths=[lora_path],
                lora_scales=[lora_scale],
                role=role,
                state_dict_transform=WanInitializer._transform_wan_lora_state_dict,
            )
        # The SVI high-noise LoRA is part of the expert's identity (0103): a
        # reloaded high expert without it would denoise under the WRONG
        # convention for the rest of the run. Re-fuse it last, matching the
        # init-time application order, under the same strict-match contract.
        for report in getattr(model, "svi_lora_reports", ()) or ():
            if report.role != "high_noise_transformer":
                continue
            result = LoRALoader.load_and_apply_lora_detailed(
                lora_mapping=WanLoRAMapping.get_mapping(),
                transformer=transformer,
                lora_paths=[report.resolved_path],
                lora_scales=[WanInitializer.SVI_LORA_SCALE],
                role=report.role,
                state_dict_transform=WanInitializer._transform_wan_lora_state_dict,
            )
            WanInitializer._require_strict_svi_key_match(result.reports[0])

    @staticmethod
    def _resolve_lora_roles(
        model,
        *,
        lora_paths: list[str] | None,
        lora_target_roles: list[str] | None,
    ) -> list[str]:
        if not lora_paths:
            if lora_target_roles:
                raise LoRAApplicationError("--lora-target-roles requires --lora-paths.")
            return []

        if model.transformer_2 is None:
            if lora_target_roles is None:
                return ["transformer"] * len(lora_paths)
            if len(lora_target_roles) != len(lora_paths):
                raise LoRAApplicationError(
                    "--lora-target-roles must provide one role per LoRA file for Wan generation."
                )
            invalid = [role for role in lora_target_roles if role != "transformer"]
            if invalid:
                raise LoRAApplicationError(
                    "Wan TI2V-5B uses one transformer; valid --lora-target-roles value is only 'transformer'."
                )
            return list(lora_target_roles)

        if lora_target_roles is None:
            raise LoRAApplicationError(
                "Wan A14B LoRAs require explicit --lora-target-roles so MLX-Gen knows whether each file "
                "targets the high-noise or low-noise denoiser."
            )
        if len(lora_target_roles) != len(lora_paths):
            raise LoRAApplicationError("--lora-target-roles must provide one role per LoRA file for Wan generation.")

        valid_roles = {"high_noise_transformer", "low_noise_transformer"}
        invalid = [role for role in lora_target_roles if role not in valid_roles]
        if invalid:
            raise LoRAApplicationError(
                "Wan A14B valid --lora-target-roles values are 'high_noise_transformer' and 'low_noise_transformer'."
            )
        return list(lora_target_roles)

    @staticmethod
    def _transformer_for_role(model, role: str) -> WanTransformer:
        if role == "transformer":
            return model.transformer
        if role == "high_noise_transformer":
            return model.transformer
        if role == "low_noise_transformer":
            if model.transformer_2 is None:
                raise LoRAApplicationError("Selected Wan model does not expose a low-noise transformer.")
            return model.transformer_2
        raise LoRAApplicationError(f"Unsupported Wan LoRA target role: {role}.")

    @staticmethod
    def _transform_wan_lora_state_dict(
        weights: dict[str, mx.array], transformer: WanTransformer
    ) -> dict[str, mx.array]:
        if not WanInitializer._transformer_uses_image_projections(transformer):
            return weights
        if WanInitializer._state_dict_has_image_projection_lora(weights):
            return weights
        expanded = dict(weights)
        WanInitializer._expand_t2v_lora_for_i2v(expanded)
        return expanded

    @staticmethod
    def _transformer_uses_image_projections(transformer: WanTransformer) -> bool:
        if not transformer.blocks:
            return False
        return getattr(transformer.blocks[0].attn2, "add_k_proj", None) is not None

    @staticmethod
    def _state_dict_has_image_projection_lora(weights: dict[str, mx.array]) -> bool:
        image_markers = ("add_k_proj", "add_v_proj", "k_img", "v_img", "cross_attn_k_img", "cross_attn_v_img")
        return any(any(marker in key for marker in image_markers) for key in weights)

    @staticmethod
    def _expand_t2v_lora_for_i2v(weights: dict[str, mx.array]) -> None:
        reference_pairs = [
            (
                "transformer.blocks.",
                ".attn2.to_k.lora_A.weight",
                ".attn2.to_k.lora_B.weight",
                ".attn2.add_k_proj.lora_A.weight",
                ".attn2.add_k_proj.lora_B.weight",
                ".attn2.add_v_proj.lora_A.weight",
                ".attn2.add_v_proj.lora_B.weight",
            ),
            (
                "blocks.",
                ".attn2.to_k.lora_A.weight",
                ".attn2.to_k.lora_B.weight",
                ".attn2.add_k_proj.lora_A.weight",
                ".attn2.add_k_proj.lora_B.weight",
                ".attn2.add_v_proj.lora_A.weight",
                ".attn2.add_v_proj.lora_B.weight",
            ),
            (
                "diffusion_model.blocks.",
                ".cross_attn.k.lora_A.weight",
                ".cross_attn.k.lora_B.weight",
                ".cross_attn.k_img.lora_A.weight",
                ".cross_attn.k_img.lora_B.weight",
                ".cross_attn.v_img.lora_A.weight",
                ".cross_attn.v_img.lora_B.weight",
            ),
            (
                "diffusion_model.blocks.",
                ".cross_attn.k.lora_down.weight",
                ".cross_attn.k.lora_up.weight",
                ".cross_attn.k_img.lora_down.weight",
                ".cross_attn.k_img.lora_up.weight",
                ".cross_attn.v_img.lora_down.weight",
                ".cross_attn.v_img.lora_up.weight",
            ),
            (
                "lora_unet_blocks_",
                "_cross_attn_k.lora_A.weight",
                "_cross_attn_k.lora_B.weight",
                "_cross_attn_k_img.lora_A.weight",
                "_cross_attn_k_img.lora_B.weight",
                "_cross_attn_v_img.lora_A.weight",
                "_cross_attn_v_img.lora_B.weight",
            ),
            (
                "lora_unet_blocks_",
                "_cross_attn_k.lora_down.weight",
                "_cross_attn_k.lora_up.weight",
                "_cross_attn_k_img.lora_down.weight",
                "_cross_attn_k_img.lora_up.weight",
                "_cross_attn_v_img.lora_down.weight",
                "_cross_attn_v_img.lora_up.weight",
            ),
        ]

        for (
            prefix,
            ref_a_suffix,
            ref_b_suffix,
            add_k_a_suffix,
            add_k_b_suffix,
            add_v_a_suffix,
            add_v_b_suffix,
        ) in reference_pairs:
            WanInitializer._expand_projection_family(
                weights,
                prefix=prefix,
                ref_a_suffix=ref_a_suffix,
                ref_b_suffix=ref_b_suffix,
                add_k_a_suffix=add_k_a_suffix,
                add_k_b_suffix=add_k_b_suffix,
                add_v_a_suffix=add_v_a_suffix,
                add_v_b_suffix=add_v_b_suffix,
            )

    @staticmethod
    def _expand_projection_family(
        weights: dict[str, mx.array],
        *,
        prefix: str,
        ref_a_suffix: str,
        ref_b_suffix: str,
        add_k_a_suffix: str,
        add_k_b_suffix: str,
        add_v_a_suffix: str,
        add_v_b_suffix: str,
    ) -> None:
        prefixes = [
            key[: -len(ref_a_suffix)]
            for key in list(weights.keys())
            if key.startswith(prefix) and key.endswith(ref_a_suffix)
        ]
        for key_prefix in prefixes:
            ref_a = f"{key_prefix}{ref_a_suffix}"
            ref_b = f"{key_prefix}{ref_b_suffix}"
            if ref_a not in weights or ref_b not in weights:
                continue
            weights.setdefault(f"{key_prefix}{add_k_a_suffix}", mx.zeros_like(weights[ref_a]))
            weights.setdefault(f"{key_prefix}{add_k_b_suffix}", mx.zeros_like(weights[ref_b]))
            weights.setdefault(f"{key_prefix}{add_v_a_suffix}", mx.zeros_like(weights[ref_a]))
            weights.setdefault(f"{key_prefix}{add_v_b_suffix}", mx.zeros_like(weights[ref_b]))

    @staticmethod
    def _validate_component_quantization_layout(
        component_name: str, component_weights: dict, q_level: int | None
    ) -> None:
        if q_level != 8 or component_name not in ("transformer", "transformer_2"):
            return

        incompatible_paths = WanInitializer._q8_sensitive_weight_paths(component_weights)
        if not incompatible_paths:
            return

        examples = ", ".join(incompatible_paths[:3])
        if len(incompatible_paths) > 3:
            examples += ", ..."
        raise ValueError(
            "Wan q8 checkpoint uses an incompatible older quantization layout: "
            f"{component_name} stores q8 tensors for BF16-only paths ({examples}). "
            "Regenerate or re-download the checkpoint with the current Wan q8 policy; "
            "conditioning and output projection layers must remain BF16."
        )

    @staticmethod
    def _normalize_runtime_sensitive_q8_paths(
        component_name: str,
        component_weights: dict,
        q_level: int | None,
    ) -> None:
        if q_level != 8 or component_name not in ("transformer", "transformer_2"):
            return

        normalized_paths: list[str] = []
        WanInitializer._normalize_runtime_sensitive_q8_paths_recursive(
            node=component_weights,
            path="",
            normalized_paths=normalized_paths,
        )
        if normalized_paths:
            preview = ", ".join(normalized_paths[:3])
            if len(normalized_paths) > 3:
                preview += ", ..."
            print(f"⚠️  Normalizing Wan q8 runtime-sensitive paths to BF16 at load: {preview}")

    @staticmethod
    def _normalize_runtime_sensitive_q8_paths_recursive(
        node,
        *,
        path: str,
        normalized_paths: list[str],
    ) -> None:
        if isinstance(node, list):
            for index, value in enumerate(node):
                next_path = f"{path}.{index}" if path else str(index)
                WanInitializer._normalize_runtime_sensitive_q8_paths_recursive(
                    value,
                    path=next_path,
                    normalized_paths=normalized_paths,
                )
            return

        if not isinstance(node, dict):
            return

        if WanInitializer._is_quantized_linear_state(node) and WanInitializer._is_runtime_sensitive_q8_path(path):
            node["weight"] = WanInitializer._dequantized_linear_weight(node)
            node.pop("scales", None)
            node.pop("biases", None)
            normalized_paths.append(path)
            return

        for key, value in node.items():
            next_path = f"{path}.{key}" if path else str(key)
            WanInitializer._normalize_runtime_sensitive_q8_paths_recursive(
                value,
                path=next_path,
                normalized_paths=normalized_paths,
            )

    @staticmethod
    def _is_quantized_linear_state(node: dict) -> bool:
        return {
            "weight",
            "scales",
            "biases",
        }.issubset(node.keys())

    @staticmethod
    def _dequantized_linear_weight(node: dict) -> mx.array:
        bits = 8
        input_dims = int(node["weight"].shape[1]) * (32 // bits)
        scale_columns = int(node["scales"].shape[1])
        if scale_columns <= 0 or input_dims % scale_columns != 0:
            raise ValueError(
                "Cannot infer Wan q8 group size for runtime normalization: "
                f"weight={tuple(node['weight'].shape)}, scales={tuple(node['scales'].shape)}."
            )
        group_size = input_dims // scale_columns
        return mx.dequantize(
            node["weight"],
            node["scales"],
            node["biases"],
            group_size=group_size,
            bits=bits,
        ).astype(ModelConfig.precision)

    @staticmethod
    def _is_runtime_sensitive_q8_path(path: str) -> bool:
        return path.endswith(
            (
                ".attn1.to_q",
                ".attn1.to_k",
                ".attn1.to_v",
                ".attn1.to_out.0",
                ".attn2.to_q",
                ".attn2.to_k",
                ".attn2.to_v",
                ".attn2.to_out.0",
                ".attn2.add_k_proj",
                ".attn2.add_v_proj",
                ".ffn.net.0",
                ".ffn.net.1",
            )
        )

    @staticmethod
    def _q8_sensitive_weight_paths(weights: dict, prefix: str = "") -> list[str]:
        paths = []
        for key, value in weights.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                if ("scales" in value or "biases" in value) and WanWeightDefinition._is_q8_sensitive_transformer_path(
                    path
                ):
                    paths.append(path)
                paths.extend(WanInitializer._q8_sensitive_weight_paths(value, path))
        return paths

    @staticmethod
    def _transformer_kwargs(model_config: ModelConfig) -> dict:
        allowed = {
            "patch_size",
            "num_attention_heads",
            "attention_head_dim",
            "in_channels",
            "out_channels",
            "text_dim",
            "freq_dim",
            "ffn_dim",
            "num_layers",
            "cross_attn_norm",
            "eps",
            "added_kv_proj_dim",
            "rope_max_seq_len",
            "vace_layers",
            "vace_in_channels",
        }
        kwargs = {key: value for key, value in model_config.transformer_overrides.items() if key in allowed}
        if "patch_size" in kwargs:
            kwargs["patch_size"] = tuple(kwargs["patch_size"])
        return kwargs

    @staticmethod
    def _vae_kwargs(model_config: ModelConfig) -> dict:
        return dict(model_config.transformer_overrides.get("vae_config", {}))

    @staticmethod
    def _resolve_component_sources(
        *,
        model_config: ModelConfig,
        model_path: str | None,
        weight_definition: WanWeightDefinition,
    ) -> _WanComponentSources:
        if model_path is not None:
            return WanInitializer._resolve_monolithic_source(
                source=model_path,
                weight_definition=weight_definition,
            )

        configured_base = model_config.transformer_overrides.get("component_base_model")
        if configured_base is not None and (not isinstance(configured_base, str) or not configured_base.strip()):
            raise ValueError("Wan component_base_model must be a non-empty repository id or local path.")
        base_source = configured_base or model_config.base_model
        transformer_source = model_config.custom_transformer_model

        if configured_base is not None and not transformer_source:
            raise ValueError(
                "Wan factored component sources require custom_transformer_model when component_base_model is set."
            )
        if base_source and transformer_source:
            return WanInitializer._resolve_factored_sources(
                base_source=base_source,
                transformer_source=transformer_source,
                model_config=model_config,
                weight_definition=weight_definition,
            )
        return WanInitializer._resolve_monolithic_source(
            source=model_config.model_name,
            weight_definition=weight_definition,
        )

    @staticmethod
    def _resolve_monolithic_source(
        *,
        source: str,
        weight_definition: WanWeightDefinition,
    ) -> _WanComponentSources:
        root_path = PathResolution.resolve(path=source, patterns=weight_definition.get_download_patterns())
        if root_path is None:
            raise FileNotFoundError(f"Wan model source {source!r} did not resolve to a model root.")
        component_names = [component.name for component in weight_definition.get_components()]
        component_names.extend(["text_encoder", "tokenizer"])
        component_roots = {name: root_path for name in component_names}
        provenance = {
            name: WanInitializer._component_provenance(source=source, root_path=root_path, source_role="monolithic")
            for name in component_names
        }
        return _WanComponentSources(
            root_path=root_path,
            component_roots=component_roots,
            provenance=provenance,
            factored=False,
        )

    @staticmethod
    def _resolve_factored_sources(
        *,
        base_source: str,
        transformer_source: str,
        model_config: ModelConfig,
        weight_definition: WanWeightDefinition,
    ) -> _WanComponentSources:
        base_patterns = weight_definition.get_base_download_patterns()
        transformer_patterns = weight_definition.get_transformer_download_patterns()
        base_revision = model_config.transformer_overrides.get("expected_component_base_revision")
        transformer_revision_expected = model_config.transformer_overrides.get("expected_transformer_revision")
        base_root = PathResolution.resolve(path=base_source, patterns=base_patterns, revision=base_revision)
        if base_root is None:
            raise FileNotFoundError(f"Wan base component source {base_source!r} did not resolve to a model root.")
        WanInitializer._require_source_patterns(
            source=base_source,
            root_path=base_root,
            patterns=base_patterns,
            source_role="base components",
        )
        WanInitializer._validate_expected_snapshot_revision(
            source=base_source,
            root_path=base_root,
            expected_revision=base_revision,
            source_role="base component",
        )
        transformer_root = PathResolution.resolve(
            path=transformer_source,
            patterns=transformer_patterns,
            revision=transformer_revision_expected,
        )
        if transformer_root is None:
            raise FileNotFoundError(
                f"Wan transformer component source {transformer_source!r} did not resolve to a model root."
            )
        WanInitializer._require_source_patterns(
            source=transformer_source,
            root_path=transformer_root,
            patterns=transformer_patterns,
            source_role="transformer",
        )

        WanInitializer._validate_expected_snapshot_revision(
            source=transformer_source,
            root_path=transformer_root,
            expected_revision=transformer_revision_expected,
            source_role="transformer",
        )

        component_roots = {
            "text_encoder": base_root,
            "tokenizer": base_root,
            "vae": base_root,
        }
        for component in weight_definition.get_components():
            component_roots[component.name] = (
                transformer_root if component.name.startswith("transformer") else base_root
            )
        provenance = {
            name: WanInitializer._component_provenance(
                source=base_source,
                root_path=base_root,
                source_role="base",
            )
            for name in ("text_encoder", "tokenizer", "vae")
        }
        for component in weight_definition.get_components():
            source = transformer_source if component.name.startswith("transformer") else base_source
            root_path = transformer_root if component.name.startswith("transformer") else base_root
            source_role = "transformer" if component.name.startswith("transformer") else "base"
            provenance[component.name] = WanInitializer._component_provenance(
                source=source,
                root_path=root_path,
                source_role=source_role,
            )
        return _WanComponentSources(
            root_path=base_root,
            component_roots=component_roots,
            provenance=provenance,
            factored=True,
        )

    @staticmethod
    def _require_source_patterns(
        *,
        source: str,
        root_path: Path,
        patterns: list[str],
        source_role: str,
    ) -> None:
        required_subdirs = PathResolution._get_required_subdirs_with_safetensors(patterns)
        if PathResolution._is_snapshot_complete(root_path, required_subdirs, patterns):
            return
        raise FileNotFoundError(
            f"Wan factored {source_role} source {source!r} resolved to incomplete root {root_path}. "
            f"Required patterns: {patterns}. MLX-Gen will not substitute another cached model."
        )

    @staticmethod
    def _component_provenance(*, source: str, root_path: Path, source_role: str) -> dict[str, str]:
        provenance = {
            "source": source,
            "source_role": source_role,
        }
        revision = WanInitializer._snapshot_revision(root_path)
        if revision is not None:
            provenance["revision"] = revision
        return provenance

    @staticmethod
    def _validate_expected_snapshot_revision(
        *,
        source: str,
        root_path: Path,
        expected_revision: str | None,
        source_role: str,
    ) -> None:
        if expected_revision is None:
            return
        actual_revision = WanInitializer._snapshot_revision(root_path)
        if actual_revision == expected_revision:
            return
        actual_label = actual_revision if actual_revision is not None else "unverifiable local path"
        raise ValueError(
            f"Wan {source_role} revision mismatch: {source!r} resolved to {actual_label!r}, "
            f"but the selected model config requires {expected_revision!r}."
        )

    @staticmethod
    def _snapshot_revision(root_path: Path) -> str | None:
        parts = root_path.parts
        for index, part in enumerate(parts[:-1]):
            if part == "snapshots":
                return parts[index + 1]
        return None

    @staticmethod
    def _validate_factored_source_config(
        *,
        base_root: Path,
        transformer_root: Path,
        model_config: ModelConfig,
    ) -> None:
        overrides = model_config.transformer_overrides
        expected_renderer_config = overrides.get("expected_renderer_config")
        if isinstance(expected_renderer_config, dict):
            renderer_config = WanInitializer._read_json(transformer_root / "config.json")
            if renderer_config is None:
                raise FileNotFoundError(f"Wan factored Bernini source {transformer_root} has no top-level config.json.")
            for key in expected_renderer_config:
                WanInitializer._validate_config_key(
                    transformer_root,
                    renderer_config,
                    expected_renderer_config,
                    f"renderer.{key}",
                    key,
                )
        transformer_config = WanInitializer._read_json(transformer_root / "transformer" / "config.json")
        if transformer_config is None:
            raise FileNotFoundError(
                f"Wan factored transformer source {transformer_root} has no transformer/config.json."
            )
        WanInitializer._validate_transformer_config(
            transformer_root,
            transformer_config,
            overrides,
            "transformer",
        )

        transformer_2_config = WanInitializer._read_json(transformer_root / "transformer_2" / "config.json")
        has_transformer_2 = bool(overrides.get("has_transformer_2", False))
        if has_transformer_2 and transformer_2_config is None:
            raise FileNotFoundError(
                f"Wan factored transformer source {transformer_root} has no transformer_2/config.json."
            )
        if transformer_2_config is not None and not has_transformer_2:
            WanInitializer._raise_source_mismatch(
                root_path=transformer_root,
                key="transformer_2",
                actual="present",
                expected="absent",
            )
        if transformer_2_config is not None:
            WanInitializer._validate_transformer_config(
                transformer_root,
                transformer_2_config,
                overrides,
                "transformer_2",
            )

        vae_config = WanInitializer._read_json(base_root / "vae" / "config.json")
        if vae_config is None:
            raise FileNotFoundError(f"Wan factored base source {base_root} has no vae/config.json.")
        WanInitializer._validate_vae_config(base_root, vae_config, overrides)

        text_encoder_config = WanInitializer._read_json(base_root / "text_encoder" / "config.json")
        if text_encoder_config is None:
            raise FileNotFoundError(f"Wan factored base source {base_root} has no text_encoder/config.json.")
        WanInitializer._validate_text_encoder_config(
            base_root,
            text_encoder_config,
            model_config.text_encoder_overrides,
        )

        tokenizer_config = WanInitializer._read_json(base_root / "tokenizer" / "tokenizer_config.json")
        tokenizer_json = WanInitializer._read_json(base_root / "tokenizer" / "tokenizer.json")
        if (
            tokenizer_config is None
            or tokenizer_json is None
            or not (base_root / "tokenizer" / "spiece.model").exists()
        ):
            raise FileNotFoundError(
                f"Wan factored base source {base_root} needs tokenizer_config.json, tokenizer.json, and spiece.model."
            )
        WanInitializer._validate_tokenizer_config(
            base_root,
            tokenizer_config,
            tokenizer_json,
            model_config.text_encoder_overrides,
        )
        WanInitializer._validate_factored_scheduler_semantics(base_root, overrides)

    @staticmethod
    def _validate_source_config(root_path: Path | None, model_config: ModelConfig) -> None:
        if root_path is None:
            return

        overrides = model_config.transformer_overrides
        model_index = WanInitializer._read_json(root_path / "model_index.json")
        if model_index is not None:
            WanInitializer._validate_model_index(root_path, model_index, overrides)

        transformer_config = WanInitializer._read_json(root_path / "transformer" / "config.json")
        if transformer_config is not None:
            WanInitializer._validate_transformer_config(root_path, transformer_config, overrides, "transformer")

        transformer_2_config = WanInitializer._read_json(root_path / "transformer_2" / "config.json")
        has_transformer_2 = bool(overrides.get("has_transformer_2", False))
        if transformer_2_config is not None and not has_transformer_2:
            raise ValueError(
                "Wan source/config mismatch: "
                f"{root_path} contains transformer_2/config.json, but {model_config.model_name} is configured "
                "as a single-transformer Wan model."
            )
        if transformer_2_config is not None:
            WanInitializer._validate_transformer_config(root_path, transformer_2_config, overrides, "transformer_2")

        vae_config = WanInitializer._read_json(root_path / "vae" / "config.json")
        if vae_config is not None:
            WanInitializer._validate_vae_config(root_path, vae_config, overrides)

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        if not path.exists():
            return None
        with path.open("rt") as json_file:
            return json.load(json_file)

    @staticmethod
    def _validate_model_index(root_path: Path, model_index: dict, overrides: dict) -> None:
        expected_has_transformer_2 = bool(overrides.get("has_transformer_2", False))
        actual_transformer_2 = model_index.get("transformer_2")
        actual_has_transformer_2 = isinstance(actual_transformer_2, list) and any(
            value is not None for value in actual_transformer_2
        )
        if actual_transformer_2 is not None and actual_has_transformer_2 != expected_has_transformer_2:
            WanInitializer._raise_source_mismatch(
                root_path=root_path,
                key="model_index.transformer_2",
                actual=actual_transformer_2,
                expected="present" if expected_has_transformer_2 else "absent",
            )

        for key in ("boundary_ratio", "expand_timesteps"):
            if key in model_index and key in overrides and model_index[key] != overrides[key]:
                WanInitializer._raise_source_mismatch(
                    root_path=root_path,
                    key=f"model_index.{key}",
                    actual=model_index[key],
                    expected=overrides[key],
                )

    @staticmethod
    def _validate_transformer_config(
        root_path: Path, transformer_config: dict, overrides: dict, component: str
    ) -> None:
        for key in (
            "in_channels",
            "out_channels",
            "num_layers",
            "num_attention_heads",
            "attention_head_dim",
            "ffn_dim",
        ):
            WanInitializer._validate_config_key(root_path, transformer_config, overrides, f"{component}.{key}", key)
        WanInitializer._validate_config_key(
            root_path, transformer_config, overrides, f"{component}.patch_size", "patch_size"
        )

    @staticmethod
    def _validate_vae_config(root_path: Path, vae_config: dict, overrides: dict) -> None:
        expected_vae_config = overrides.get("vae_config", {})
        for key in (
            "base_dim",
            "decoder_base_dim",
            "z_dim",
            "in_channels",
            "out_channels",
            "patch_size",
            "scale_factor_spatial",
            "scale_factor_temporal",
            "is_residual",
        ):
            WanInitializer._validate_config_key(root_path, vae_config, expected_vae_config, f"vae.{key}", key)

    @staticmethod
    def _validate_text_encoder_config(root_path: Path, source: dict, expected: dict) -> None:
        for key in ("model_type", "d_model", "d_ff", "num_layers", "num_heads", "vocab_size"):
            WanInitializer._validate_config_key(root_path, source, expected, f"text_encoder.{key}", key)

    @staticmethod
    def _validate_tokenizer_config(
        root_path: Path,
        tokenizer_config: dict,
        tokenizer_json: dict,
        text_encoder_config: dict,
    ) -> None:
        tokenizer_class = tokenizer_config.get("tokenizer_class")
        if tokenizer_class not in {"T5Tokenizer", "T5TokenizerFast"}:
            WanInitializer._raise_source_mismatch(
                root_path=root_path,
                key="tokenizer.tokenizer_class",
                actual=tokenizer_class,
                expected="T5Tokenizer or T5TokenizerFast",
            )
        vocab = tokenizer_json.get("model", {}).get("vocab")
        vocab_size = len(vocab) if isinstance(vocab, list) else None
        encoder_vocab_size = text_encoder_config.get("vocab_size")
        if vocab_size is None or not isinstance(encoder_vocab_size, int) or vocab_size > encoder_vocab_size:
            WanInitializer._raise_source_mismatch(
                root_path=root_path,
                key="tokenizer.vocab_size",
                actual=vocab_size,
                expected=f"at most text encoder vocab_size {encoder_vocab_size!r}",
            )

    @staticmethod
    def _validate_factored_scheduler_semantics(root_path: Path, overrides: dict) -> None:
        solver = overrides.get("default_solver")
        flow_shift = overrides.get("flow_shift")
        if solver != "unipc":
            WanInitializer._raise_source_mismatch(
                root_path=root_path,
                key="scheduler.default_solver",
                actual=solver,
                expected="unipc",
            )
        if not isinstance(flow_shift, (int, float)) or not math.isfinite(float(flow_shift)) or flow_shift <= 0:
            WanInitializer._raise_source_mismatch(
                root_path=root_path,
                key="scheduler.flow_shift",
                actual=flow_shift,
                expected="a positive finite code-native UniPC flow shift",
            )

    @staticmethod
    def _validate_config_key(root_path: Path, source: dict, expected_source: dict, label: str, key: str) -> None:
        if key not in source or key not in expected_source:
            return
        if source[key] != expected_source[key]:
            WanInitializer._raise_source_mismatch(
                root_path=root_path,
                key=label,
                actual=source[key],
                expected=expected_source[key],
            )

    @staticmethod
    def _raise_source_mismatch(root_path: Path, key: str, actual, expected) -> None:
        raise ValueError(
            "Wan source/config mismatch: "
            f"{root_path} has {key}={actual!r}, but the selected Wan runtime expects {expected!r}. "
            "Pass the exact Wan model/config that matches this checkpoint; MLX-Gen will not fall back silently."
        )
