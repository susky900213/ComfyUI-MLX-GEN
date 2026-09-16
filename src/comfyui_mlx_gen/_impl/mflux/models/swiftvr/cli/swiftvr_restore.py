"""Command-line entry point for SwiftVR one-step video restoration.

Reached through ``mlxgen upscale --model swiftvr``. The parser surface is the one SeedVR2
already uses, so ``--video-path``, ``--start-seconds``, ``--max-frames``, ``--drop-audio``
and ``--output`` behave identically across both restoration families; the options SwiftVR
cannot honour are rejected with a message naming the alternative rather than accepted and
ignored.

SwiftVR restores at the source resolution in a single forward pass per chunk. Chunking is
the model's own fixed ``4a + 1`` protocol, not a memory-safety decision, so there is no
mode axis to select and the preflight line reports a chunk count.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.output_paths import normalize_output_template, resolve_output_path
from mflux.cli.parser.parsers import CommandLineParser
from mflux.cli.runtime_events import CliRuntimeEventStream, cli_print, emit_cli_failure_event_for_argv
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.download_policy import DownloadRequiredError, is_huggingface_repo_id
from mflux.models.common.restore_capabilities import (
    INPUT_VIDEO,
    RestoreCapability,
    RestoreFamilyCapabilities,
    get_restore_capabilities,
    require_capability,
)
from mflux.models.seedvr2.cli.seedvr2_upscale import (
    DEFAULT_SEEDVR2_VIDEO_CACHE_LIMIT_GB,
    SeedVR2VideoRunLock,
    _expand_video_paths,
    _provided_options,
    _validate_batch_output_collisions,
)
from mflux.models.swiftvr.streaming.chunk import aligned_frame_count, build_chunk_specs
from mflux.models.swiftvr.swiftvr_initializer import SwiftVRInitializer
from mflux.models.swiftvr.variants.upscale.swiftvr import SwiftVR
from mflux.models.swiftvr.variants.upscale.swiftvr_util import SwiftVRUtil
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.scale_factor import ScaleFactor
from mflux.utils.video_util import VideoUtil

SWIFTVR_ALIASES = {"swiftvr", "swiftvr-5b"}
SWIFTVR_REPO_IDS = {"h-oliday/swiftvr"}


@dataclass(frozen=True)
class SwiftVRRestorePlan:
    """Everything decided before the model is loaded, for the preflight line and metadata.

    There is deliberately no ``restore_mode`` field. On the SeedVR2 route that name labels
    a memory-routing decision paired with a ``route_reason``; SwiftVR has no such axis, so
    reusing the key would assert a mode that does not exist.
    """

    requested_frames: int
    aligned_frames: int
    target_height: int
    target_width: int
    padded_height: int
    padded_width: int
    clip_len: int
    dit_overlap: int
    chunk_count: int
    route_reason: str
    warnings: tuple[str, ...]


def resolve_swiftvr_model(model_arg: str | None, model_path: str | None) -> tuple[ModelConfig, str | None]:
    """Resolve a SwiftVR handle to its catalog entry and checkpoint path.

    Raises:
        ValueError: If the handle is not one SwiftVR recognises.
    """
    model_config = ModelConfig.swiftvr()
    if model_arg is None:
        return model_config, model_path

    normalized = model_arg.strip().lower()
    if normalized in SWIFTVR_ALIASES:
        return model_config, model_path
    if normalized in SWIFTVR_REPO_IDS:
        return model_config, model_path if model_path is not None else model_arg

    candidate = model_path if model_path is not None else model_arg
    # `Path("").is_dir()` is True - it resolves to the working directory - so an empty
    # handle would resolve to "load a SwiftVR checkpoint from wherever you happen to be".
    if candidate and candidate.strip() and Path(candidate).expanduser().is_dir():
        return model_config, candidate

    if is_huggingface_repo_id(model_arg):
        raise ValueError(
            f"Unsupported SwiftVR model handle {model_arg!r}. Use swiftvr, swiftvr-5b, "
            "H-oliday/SwiftVR, or an explicit local SwiftVR path."
        )
    raise ValueError(
        f"SwiftVR could not resolve {model_arg!r} to a checkpoint. Use swiftvr, swiftvr-5b, "
        "H-oliday/SwiftVR, or a local directory holding reae.safetensors and transformer/."
    )


def _plan_swiftvr_restore(
    *,
    capability: RestoreCapability,
    source_probe,
    requested_frames: int,
    resolution: int | ScaleFactor,
    clip_len: int,
    dit_overlap: int,
    force_unsafe_memory_profile: bool,
) -> SwiftVRRestorePlan:
    """Resolve geometry and the chunk plan before any weights are loaded.

    The frame ceiling comes from ``capability``, which derives it from the catalog's
    rotary table, rather than from a second copy of that arithmetic here. The route keeps
    its own ``_assert_clip_length_supported`` guard: it is a public Python API and must
    fail closed without the CLI.

    Raises:
        ValueError: If the request is outside the supported envelope.
    """
    # `aligned_frame_count` already refuses anything below one frame, so the only way
    # this predicate can fail is the declared ceiling.
    aligned_frames = aligned_frame_count(requested_frames)
    if not capability.allows_source_frame_count(aligned_frames):
        raise ValueError(
            f"SwiftVR can restore at most {capability.max_source_frames} source frames in one run, but this "
            f"request covers {aligned_frames}. Split the clip with --start-seconds and --max-frames, then "
            "join the outputs."
        )

    target_height, target_width = SwiftVRUtil.output_canvas(
        source_width=source_probe.source_width,
        source_height=source_probe.source_height,
        resolution=resolution,
    )
    padded_height, padded_width = SwiftVRUtil.padded_canvas(target_height, target_width)
    chunk_specs = build_chunk_specs(aligned_frames, clip_len)

    warnings: list[str] = []
    canvas_error = SwiftVRUtil.canvas_bound_error(padded_height, padded_width)
    if canvas_error is not None:
        warnings.append(canvas_error)
    if aligned_frames != requested_frames:
        warnings.append(
            f"SwiftVR trimmed {requested_frames} requested frames to {aligned_frames}: the chunk protocol "
            "requires a clip length of the form 4a + 1."
        )

    return SwiftVRRestorePlan(
        requested_frames=requested_frames,
        aligned_frames=aligned_frames,
        target_height=target_height,
        target_width=target_width,
        padded_height=padded_height,
        padded_width=padded_width,
        clip_len=clip_len,
        dit_overlap=dit_overlap,
        chunk_count=len(chunk_specs),
        route_reason="unsafe_override" if force_unsafe_memory_profile else "bounded_offline",
        warnings=tuple(warnings),
    )


def _print_swiftvr_preflight(video_path: Path, source_probe, plan: SwiftVRRestorePlan, *, json_events: bool) -> None:
    """One-line summary of the resolved plan. Reports chunks, never a mode."""
    cli_print(
        "SwiftVR video preflight: "
        f"source={source_probe.source_width}x{source_probe.source_height} "
        f"target={plan.target_width}x{plan.target_height} "
        f"pad={plan.padded_width}x{plan.padded_height} "
        f"frames={plan.aligned_frames} "
        f"chunks={plan.chunk_count} "
        f"clip_len={plan.clip_len} "
        f"dit_overlap={plan.dit_overlap} "
        f"reason={plan.route_reason} "
        f"video={video_path.name}",
        json_events=json_events,
    )
    for warning in plan.warnings:
        cli_print(f"SwiftVR warning: {warning}", json_events=json_events)


# How argparse reveals that an option was used. Three shapes are needed and none of them
# belong in the capability record, which must not learn argparse: a truthy attribute, a
# not-None attribute, and a count. Everything else is answered by the option token simply
# appearing on the command line, which is what the fallback below checks.
_OPTION_DETECTORS = {
    "--image-path": lambda args, provided: bool(args.image_path),
    "--quantize": lambda args, provided: args.quantize is not None,
    "--vae-tiling": lambda args, provided: bool(args.vae_tiling),
    "--stepwise-image-output-dir": lambda args, provided: bool(getattr(args, "stepwise_image_output_dir", None)),
    "--seed": lambda args, provided: len(args.seed) > 1,
}


def _reject_unsupported_options(
    parser: CommandLineParser,
    args,
    provided: set[str],
    capabilities: RestoreFamilyCapabilities,
) -> None:
    """Reject flags SwiftVR cannot honour, naming what to use instead.

    The messages are the family's declared contract rather than literals typed here, so
    the CLI and the capability record cannot drift into disagreeing about what SwiftVR
    refuses. Iteration order is the record's insertion order, which preserves the order
    the old if-chain reported in: a command breaking several rules names the same first
    problem it always did.
    """
    for option, message in capabilities.unsupported_options.items():
        detect = _OPTION_DETECTORS.get(option) or (lambda args, provided, token=option: token in provided)
        if detect(args, provided):
            # ``replace`` rather than ``format``: a message containing a literal brace
            # would otherwise raise here, which is the worst possible moment for it.
            parser.error(message.replace("{value}", str(args.quantize)))


def _load_swiftvr_model(
    *,
    parser: CommandLineParser,
    args,
    resolved_model_path: str | None,
    model_config: ModelConfig,
) -> SwiftVR:
    """Construct the model, turning a download requirement into a clean CLI failure."""
    try:
        return SwiftVR(
            quantize=args.quantize,
            model_path=resolved_model_path,
            model_config=model_config,
        )
    except DownloadRequiredError as exc:
        if getattr(args, "json_events", False):
            emit_cli_failure_event_for_argv(
                prog=parser.prog,
                argv=sys.argv[1:],
                error=exc,
                output_path=args.output,
            )
            cli_print(str(exc), json_events=True, error=True)
            raise SystemExit(1) from None
        parser.error(str(exc))


def _run_swiftvr_restore(
    *,
    parser: CommandLineParser,
    args,
    resolved_model_path: str | None,
    model_config: ModelConfig,
    video_path: Path,
    source_probe,
    plan: SwiftVRRestorePlan,
    output_pattern: str,
    seed: int,
) -> None:
    """Load a fresh model for one clip, restore it, and release the weights."""
    model = _load_swiftvr_model(
        parser=parser,
        args=args,
        resolved_model_path=resolved_model_path,
        model_config=model_config,
    )
    memory_saver = CallbackManager.register_callbacks(args=args, model=model, latent_creator=None)
    events = CliRuntimeEventStream(
        enabled=bool(args.json_events),
        command="mlxgen upscale",
        model=model.model_config.model_name,
        seed=seed,
    )
    unsubscribe = None
    try:
        output_path = resolve_output_path(
            output_pattern,
            overwrite=args.replace,
            seed=seed,
            input_name=video_path.stem,
        )
        events.set_output_path(output_path)
        unsubscribe = events.subscribe_model(
            model,
            map_complete_to_generated=False,
            suppress_terminal_phases={"failed"},
        )
        _print_swiftvr_preflight(video_path, source_probe, plan, json_events=bool(args.json_events))
        if source_probe.audio_present and not args.drop_audio:
            cli_print(
                "SwiftVR note: source audio detected; the CLI will preserve the matching source audio "
                "segment. If that cannot be proven safe, the run fails. Pass --drop-audio to allow a "
                "silent output intentionally.",
                json_events=bool(args.json_events),
            )
        try:
            result_path = model.restore_video_to_path(
                video_path=video_path,
                resolution=args.resolution,
                output_path=output_path,
                clip_len=plan.clip_len,
                dit_overlap=plan.dit_overlap,
                start_seconds=args.start_seconds,
                max_frames=args.max_frames,
                color_correction_mode=args.color_correction,
                drop_audio=args.drop_audio,
                export_json_metadata=args.metadata,
                overwrite=True,
                validate_health=not args.no_validate_health,
                restore_metadata={
                    "restore_family": "swiftvr",
                    "requested_seed": seed,
                    "swiftvr_route_reason": plan.route_reason,
                    "mlx_cache_limit_gb": args.mlx_cache_limit_gb,
                },
                enforce_memory_budget=plan.route_reason != "unsafe_override",
            )
        except Exception as exc:
            events.emit_failed(task="video-to-video", error=exc)
            raise
        events.set_output_path(result_path)
        cli_print(f"Video saved successfully at: {result_path}", json_events=bool(args.json_events))
    finally:
        if unsubscribe is not None:
            unsubscribe()
        if memory_saver:
            cli_print(memory_saver.memory_stats(), json_events=bool(args.json_events))
        del model
        mx.clear_cache()


def main() -> None:
    """Restore videos with SwiftVR one-step chunked restoration."""
    parser = CommandLineParser(description="Restore a video with SwiftVR one-step chunked restoration.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    for action in parser._actions:
        if action.dest == "model":
            action.help = "SwiftVR model alias (swiftvr or swiftvr-5b), H-oliday/SwiftVR, or a local path."
            break
    parser.add_seedvr2_upscale_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()
    provided = _provided_options(sys.argv[1:])

    # The declared contract for this family. `upscale_main()` already picked this
    # `main()` by family, so fetching it needs no model resolution and it can run before
    # `resolve_swiftvr_model` - a rejected flag then costs nothing.
    capabilities = get_restore_capabilities(family="swiftvr")
    _reject_unsupported_options(parser, args, provided, capabilities)
    # Asked of the contract rather than assumed. When SwiftVR gains a measured image
    # route this is where the second capability is picked up, and the --image-path
    # refusal disappears on its own because the refusals are the record's.
    video_capability = require_capability(capabilities, INPUT_VIDEO)

    video_paths = _expand_video_paths(args.video_path) if args.video_path else []
    if not video_paths:
        cli_print("No videos to restore.", json_events=bool(args.json_events))
        return

    # SwiftVR restores at the source resolution, so 1x is the only meaningful default.
    if "--resolution" not in provided:
        args.resolution = ScaleFactor(1)
    # The reference pipeline writes the decoder output unchanged. The shared parser
    # defaults to wavelet for SeedVR2; leaving that in place would either fail every
    # plain SwiftVR run or silently apply an unmeasured transfer.
    if "--color-correction" not in provided:
        args.color_correction = "off"
    elif args.color_correction not in video_capability.color_correction_modes:
        # Refuse at parse time, before the 5B weight load: the route guard
        # (SwiftVR._assert_supported_options) would raise the same incapability only
        # after minutes of loading. Same wording, same alternatives.
        parser.error(
            f"SwiftVR does not apply color correction, so --color-correction {args.color_correction} "
            "cannot be honoured. The reference pipeline writes the decoder output unchanged, and "
            "MLX-Gen has not measured a color transfer against it for this model. Pass "
            "--color-correction off for SwiftVR, or use --model seedvr2-3b, whose restore path has a "
            "validated wavelet and LAB transfer."
        )
    if not args.low_ram:
        args.low_ram = True
    if args.mlx_cache_limit_gb is None:
        args.mlx_cache_limit_gb = DEFAULT_SEEDVR2_VIDEO_CACHE_LIMIT_GB

    try:
        model_config, resolved_model_path = resolve_swiftvr_model(args.model, args.model_path)
    except ValueError as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    # The chunk plan's defaults live in the catalog entry, not in this module, so the
    # preflight line and the run cannot drift apart.
    runtime_settings = SwiftVRInitializer.runtime_settings(model_config)

    video_probes: dict[Path, object] = {}
    restore_plans: dict[Path, SwiftVRRestorePlan] = {}
    try:
        for video_path in video_paths:
            source_probe = VideoUtil.read_video_clip(video_path, start_seconds=args.start_seconds, max_frames=1)
            video_probes[video_path] = source_probe
            plan = _plan_swiftvr_restore(
                capability=video_capability,
                source_probe=source_probe,
                requested_frames=VideoUtil.requested_clip_frame_count(source_probe, args.max_frames),
                resolution=args.resolution,
                clip_len=runtime_settings.clip_len,
                dit_overlap=runtime_settings.dit_overlap,
                force_unsafe_memory_profile=args.force_unsafe_video_memory,
            )
            if plan.warnings and not args.force_unsafe_video_memory:
                blocking = [w for w in plan.warnings if "--force-unsafe-video-memory" in w]
                if blocking:
                    parser.error(blocking[0])
            restore_plans[video_path] = plan
        CallbackManager.apply_runtime_memory_options(args)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        output_pattern = normalize_output_template(
            args.output,
            is_video=True,
            include_seed=False,
            include_input_name=len(video_paths) > 1,
        )
        _validate_batch_output_collisions(
            output_pattern=output_pattern,
            image_paths=[],
            video_paths=video_paths,
            seeds=args.seed,
            replace=args.replace,
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        with SeedVR2VideoRunLock():
            for video_path in video_paths:
                _run_swiftvr_restore(
                    parser=parser,
                    args=args,
                    resolved_model_path=resolved_model_path,
                    model_config=model_config,
                    video_path=video_path,
                    source_probe=video_probes[video_path],
                    plan=restore_plans[video_path],
                    output_pattern=output_pattern,
                    seed=args.seed[0],
                )
    except StopImageGenerationException as exc:
        cli_print(str(exc), json_events=bool(args.json_events))


if __name__ == "__main__":
    main()
