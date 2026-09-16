import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.outpaint_cli import emit_canvas_notices, prepare_canvas_session
from mflux.cli.output_paths import resolve_output_path
from mflux.cli.parser.parsers import CommandLineParser
from mflux.cli.runtime_events import CliRuntimeEventStream, cli_print
from mflux.models.common.config import ModelConfig
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.variants.edit.qwen_image_edit import (
    QwenImageEdit,
    QwenImageEdit as _QwenImageEditImplementation,
)
from mflux.outpaint import OutpaintSession
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using Qwen Image Edit with image conditioning.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_argument("--image-paths", type=Path, nargs="+", required=True, help="Local paths to one or more init images. For single image editing, provide one path. For multiple image editing, provide multiple paths.")  # fmt: off
    parser.add_mask_path_argument(
        help_text=(
            "Optional mask image path for localized Qwen edits. White pixels are repainted and black pixels are "
            "preserved."
        ),
    )
    parser.add_argument(
        "--reframe-padding",
        default=None,
        help=(
            "Generative reframe request: expand one source image by CSS-style "
            "top,right,bottom,left padding before edit generation."
        ),
    )
    parser.add_argument(
        "--outpaint-padding",
        "--image-outpaint-padding",
        dest="outpaint_padding",
        default=None,
        help=(
            "Expand one source image by CSS-style top,right,bottom,left padding and use an adaptive "
            "source blend when the generated source window still matches the original image."
        ),
    )
    parser.add_outpaint_pass_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()
    source_image_paths = [str(p) for p in args.image_paths]
    _validate_canvas_args(parser=parser, args=args, source_image_paths=source_image_paths)

    # 1. Load the model
    try:
        model_config = ModelConfig.from_name(args.model or "qwen-image-edit", base_model=args.base_model)
    except ModelConfigError:
        if args.model_path is None:
            raise
        model_config = ModelConfig.from_name(args.base_model or "qwen-image-edit")
    if len(source_image_paths) > 1 and not _QwenImageEditImplementation._is_edit_plus_model_config(
        model_config=model_config,
        image_paths=source_image_paths,
    ):
        parser.error(
            "Multiple Qwen edit reference images require an Edit-Plus model, such as "
            "qwen-image-edit-2509 or qwen-image-edit-2511."
        )
    if not _option_was_provided(sys.argv[1:], "--scheduler"):
        args.scheduler = "flow_match_euler_discrete"
    if args.guidance is None:
        if _QwenImageEditImplementation._is_edit_plus_model_config(
            model_config=model_config, image_paths=source_image_paths
        ):
            args.guidance = 4.0
        else:
            args.guidance = 4.0
    if not _option_was_provided(sys.argv[1:], "--steps") and _QwenImageEditImplementation._is_edit_plus_model_config(
        model_config=model_config,
        image_paths=source_image_paths,
    ):
        args.steps = 40

    CallbackManager.apply_runtime_memory_options(args)

    qwen = QwenImageEdit(
        quantize=args.quantize,
        model_config=model_config,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=qwen,
        latent_creator=QwenLatentCreator,
    )

    try:
        with TemporaryDirectory(prefix="mlxgen-outpaint-") as temporary_directory:
            # The conditioning canvas, the fill policy, the guard and the metadata are the shared
            # outpaint layer's; this command only supplies the parsed request and the model.
            try:
                canvas_session = prepare_canvas_session(
                    args=args,
                    source_image_paths=source_image_paths,
                    workspace=temporary_directory,
                    model_config=model_config,
                )
            except ValueError as exc:
                parser.error(str(exc))
            emit_canvas_notices(canvas_session)
            image_paths = (
                [str(path) for path in canvas_session.conditioning_image_paths]
                if canvas_session is not None
                else source_image_paths
            )

            try:
                for seed in args.seed:
                    events = CliRuntimeEventStream(
                        enabled=bool(args.json_events),
                        command="mlxgen generate",
                        model=model_config.model_name,
                        seed=seed,
                    )
                    # 4. Generate an image for each seed value
                    output_path = resolve_output_path(args.output, overwrite=args.replace, seed=seed)
                    events.set_output_path(output_path)
                    unsubscribe = events.subscribe_model(qwen, map_complete_to_generated=True)
                    try:
                        if isinstance(canvas_session, OutpaintSession):
                            # The session owns the canvas keywords and runs every planned pass;
                            # the per-pass geometry is not the final --width/--height.
                            image = canvas_session.generate(
                                qwen,
                                seed=seed,
                                prompt=PromptUtil.read_prompt(args),
                                negative_prompt=_read_negative_prompt(args),
                                guidance=args.guidance,
                                num_inference_steps=args.steps,
                                scheduler=args.scheduler,
                            )
                        else:
                            image = qwen.generate_image(
                                seed=seed,
                                prompt=PromptUtil.read_prompt(args),
                                negative_prompt=_read_negative_prompt(args),
                                width=args.width,
                                height=args.height,
                                guidance=args.guidance,
                                image_path=source_image_paths[0],  # Use original source for metadata
                                image_paths=image_paths,
                                mask_path=args.mask_path,
                                num_inference_steps=args.steps,
                                scheduler=args.scheduler,
                                canvas_policy=args.canvas_policy,
                            )
                            if canvas_session is not None:
                                canvas_session.finalize(image)

                        events.emit_save()
                        image.save(
                            path=output_path,
                            export_json_metadata=args.metadata,
                            overwrite=True,
                            embed_metadata=args.embed_metadata,
                        )
                        events.emit_complete()
                    except Exception as exc:
                        events.emit_failed(error=exc)
                        raise
                    finally:
                        if unsubscribe is not None:
                            unsubscribe()

            except (StopImageGenerationException, PromptFileReadError) as exc:
                cli_print(str(exc), json_events=bool(args.json_events))
    finally:
        if memory_saver:
            cli_print(memory_saver.memory_stats(), json_events=bool(args.json_events))


def _validate_canvas_args(*, parser: CommandLineParser, args, source_image_paths: list[str]) -> None:
    if args.mask_path is not None:
        if len(source_image_paths) != 1:
            parser.error("--mask-path requires exactly one --image-paths value.")
        if args.outpaint_padding is not None or args.reframe_padding is not None:
            parser.error("--mask-path cannot be combined with --reframe-padding or --outpaint-padding.")
    if _option_was_provided(sys.argv[1:], "--outpaint-passes") and args.outpaint_padding is None:
        parser.error("--outpaint-passes configures the --outpaint-padding run. Pass --outpaint-padding, or drop it.")
    if args.outpaint_padding is None and args.reframe_padding is None:
        return
    if args.outpaint_padding is not None and args.reframe_padding is not None:
        parser.error("--reframe-padding and --outpaint-padding are different workflows and cannot be used together.")
    option_name = "--outpaint-padding" if args.outpaint_padding is not None else "--reframe-padding"
    if len(source_image_paths) != 1:
        parser.error(f"{option_name} requires exactly one --image-paths value.")
    if _any_option_was_provided(sys.argv[1:], ("--width", "--height")):
        parser.error(f"{option_name} computes --width and --height from the source image; do not pass either option.")
    if _option_was_provided(sys.argv[1:], "--canvas-policy"):
        parser.error(f"{option_name} uses --canvas-policy exact-resize; do not pass --canvas-policy.")


def _read_negative_prompt(args) -> str | None:
    if _any_option_was_provided(sys.argv[1:], ("--negative-prompt", "--negative")):
        return PromptUtil.read_negative_prompt(args)
    return None


def _any_option_was_provided(argv: list[str], option_names: tuple[str, ...]) -> bool:
    return any(_option_was_provided(argv, option_name) for option_name in option_names)


def _option_was_provided(argv: list[str], option_name: str) -> bool:
    for token in argv:
        if token == option_name or token.startswith(f"{option_name}="):
            return True
    return False


if __name__ == "__main__":
    main()
