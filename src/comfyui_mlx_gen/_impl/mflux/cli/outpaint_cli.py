"""The argparse adapter for the shared outpaint layer.

Every backend CLI that expands a conditioning canvas goes through here, and this is the only
module that reads a parsed namespace on the way in. `mflux.outpaint` itself takes plain values,
so an embedding Python application never has to build an argparse namespace to get the same
canvas, the same fill policy and the same metadata a CLI run produces.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from mflux.outpaint import (
    OutpaintContract,
    OutpaintError,
    OutpaintSession,
    ReframeSession,
    prepare_outpaint,
    prepare_reframe,
)

CanvasSession = OutpaintSession | ReframeSession


def prepare_canvas_session(
    *,
    args,
    source_image_paths: Sequence[str | Path],
    workspace: str | Path,
    contract: OutpaintContract | None = None,
    capability: Any = None,
    model: str | None = None,
    model_config: Any = None,
    argv: Sequence[str] | None = None,
) -> CanvasSession | None:
    """Build the conditioning-canvas session a parsed namespace asks for, or None.

    Returns an `OutpaintSession` for `--outpaint-padding`, a `ReframeSession` for
    `--reframe-padding`, and None when neither was requested. The generation geometry the
    session computed is written back onto `args` (`width`, `height`, `canvas_policy`) because
    both options own those values and reject a caller-supplied one at parse time.

    Raises `OutpaintError` (a ValueError) so backend CLIs keep converting it with
    `parser.error(...)`.
    """
    outpaint_padding = getattr(args, "outpaint_padding", None)
    reframe_padding = getattr(args, "reframe_padding", None)
    if outpaint_padding is None and reframe_padding is None:
        return None
    option_name = "--outpaint-padding" if outpaint_padding is not None else "--reframe-padding"
    if len(source_image_paths) != 1:
        raise OutpaintError(f"{option_name} requires exactly one --image-paths value.")
    source_image = source_image_paths[0]

    if outpaint_padding is None:
        session: CanvasSession = prepare_reframe(
            source_image=source_image,
            padding=reframe_padding,
            workspace=workspace,
            option_name=option_name,
        )
    else:
        session = prepare_outpaint(
            source_image=source_image,
            padding=outpaint_padding,
            contract=contract,
            capability=capability,
            model=model,
            model_config=model_config,
            fill=getattr(args, "outpaint_fill", None),
            fill_color=getattr(args, "outpaint_fill_color", None),
            fill_color_explicit=option_was_provided(argv, "--outpaint-fill-color"),
            lora_paths=getattr(args, "lora_paths", None) or (),
            requested_lora_paths=getattr(args, "requested_lora_paths", None) or (),
            passes=getattr(args, "outpaint_passes", None),
            workspace=workspace,
            option_name=option_name,
        )
    args.width = session.width
    args.height = session.height
    args.canvas_policy = session.canvas_policy
    return session


def emit_canvas_notices(session: CanvasSession | None, *, stream: TextIO | None = None) -> None:
    """Print the warnings and the resolved-canvas notice for one prepared session.

    Warnings come first: they report a canvas the caller asked for against measured advice, and
    the notice below them states what actually runs.
    """
    if not isinstance(session, OutpaintSession):
        return
    stream = stream if stream is not None else sys.stderr
    for warning in session.warnings:
        print(warning, file=stream)
    print(session.notice, file=stream)


def option_was_provided(argv: Sequence[str] | None, option_name: str) -> bool:
    """Whether `option_name` appears in argv, in either the spaced or the `=` form."""
    tokens = sys.argv[1:] if argv is None else argv
    return any(token == option_name or token.startswith(f"{option_name}=") for token in tokens)


def any_option_was_provided(argv: Sequence[str] | None, option_names: Sequence[str]) -> bool:
    return any(option_was_provided(argv, option_name) for option_name in option_names)
