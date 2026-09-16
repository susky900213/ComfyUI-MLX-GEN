"""Family dispatch for the ``mlxgen upscale`` command.

Two restoration families share one command and one parser surface: SeedVR2, which
upscales images and videos, and SwiftVR, which restores video in one step at the source
resolution. This module decides which one a handle names, and does so with POSITIVE
SwiftVR matchers only - everything else, including an unrecognised Hugging Face repo id,
stays with SeedVR2 so that its own resolver raises its own error with its own wording.
Preserving SeedVR2's behaviour is therefore structural rather than something that has to
be re-verified: ``_resolve_seedvr2_model`` is called unchanged and is never reimplemented.

The same classifier runs twice per invocation - once over raw ``argv`` before argparse so
the command can pick a ``main()``, and once after parsing. They must never disagree, so
both call :func:`classify_restore_family` rather than repeating its rules.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mflux.models.common.config.model_config import ModelConfig

RestoreFamily = Literal["seedvr2", "swiftvr"]

# Handles that name SwiftVR. Everything else belongs to SeedVR2, whose resolver owns the
# unknown-handle failure.
SWIFTVR_HANDLES = frozenset({"swiftvr", "swiftvr-5b", "h-oliday/swiftvr"})

# Positive SeedVR2 recognition. Deliberately NOT "anything SeedVR2's resolver accepts":
# the resolver owns the unknown-handle error, so this set covers only the handles that
# name SeedVR2 outright. AbstractFramework packages are matched by prefix below.
SEEDVR2_HANDLES = frozenset(
    {
        "seedvr2",
        "seedvr2-3b",
        "seedvr2-7b",
        "seedvr2-7b-sharp",
        "seedvr2-7b-sharp-fp16",
        "bytedance-seed/seedvr2-3b",
        "bytedance-seed/seedvr2_3b",
        "bytedance-seed/seedvr2-7b",
        "bytedance-seed/seedvr2_7b",
        "numz/seedvr2_comfyui",
    }
)


def is_seedvr2_handle(model: str | None) -> bool:
    """True when ``model`` positively names SeedVR2: an alias, official repo or package.

    This recognition must win over any directory probe. At v0.29.0 a SeedVR2 alias
    ignored ``--path`` entirely, so a dispatcher that lets directory contents overrule
    an explicit alias silently substitutes another family for the one the user typed
    (ADR 0002).
    """
    normalized = model.strip().lower() if model else ""
    if normalized in SEEDVR2_HANDLES:
        return True
    return normalized.startswith("abstractframework/seedvr2-3b-") or normalized.startswith(
        "abstractframework/seedvr2-7b-"
    )


# A local SwiftVR checkpoint is recognised by the autoencoder that replaces the Wan 3D
# VAE; no other family in the catalog ships this file.
SWIFTVR_MARKER_FILE = "reae.safetensors"

# Files that identify a local SeedVR2 checkpoint, used only to detect a directory that
# claims to be both.
SEEDVR2_MARKER_FILES = (
    "seedvr2_ema_3b.pth",
    "seedvr2_ema_3b_fp16.safetensors",
    "seedvr2_ema_7b.pth",
    "seedvr2_ema_7b_fp16.safetensors",
    "seedvr2_ema_7b_sharp.pth",
)

SWIFTVR_HANDLE_HELP = "swiftvr, swiftvr-5b, H-oliday/SwiftVR, or a local SwiftVR path"


@dataclass(frozen=True)
class RestoreFamilyRoute:
    """Resolved family, catalog entry and checkpoint path for one restore request."""

    family: RestoreFamily
    model_config: ModelConfig
    model_path: str | None


def _looks_like_swiftvr_directory(candidate: str | None) -> bool:
    """Whether ``candidate`` is a local directory holding a SwiftVR checkpoint."""
    if not candidate:
        return False
    path = Path(candidate).expanduser()
    return path.is_dir() and (path / SWIFTVR_MARKER_FILE).exists()


def _looks_like_seedvr2_directory(candidate: str | None) -> bool:
    """Whether ``candidate`` is a local directory holding a SeedVR2 checkpoint."""
    if not candidate:
        return False
    path = Path(candidate).expanduser()
    if not path.is_dir():
        return False
    if any((path / marker).exists() for marker in SEEDVR2_MARKER_FILES):
        return True
    return (path / "transformer" / "model.safetensors.index.json").exists()


def classify_restore_family(model_arg: str | None, model_path: str | None) -> RestoreFamily:
    """Decide which restoration family a ``--model`` / ``--path`` pair names.

    Args:
        model_arg: The ``--model`` value, or ``None`` when it was not given.
        model_path: The ``--path`` value, or ``None``.

    Returns:
        ``"swiftvr"`` for a SwiftVR alias, repo id or local checkpoint; ``"seedvr2"``
        otherwise, including for handles neither family recognises.

    Raises:
        ValueError: If a directory holds both a SwiftVR and a SeedVR2 checkpoint, which
            no alias can disambiguate.
    """
    if model_arg is not None and model_arg.strip().lower() in SWIFTVR_HANDLES:
        return "swiftvr"
    if is_seedvr2_handle(model_arg):
        # An explicit SeedVR2 alias must win over directory contents: at v0.29.0 the
        # alias branch ignored --path entirely, and letting a SwiftVR marker file
        # overrule the handle the user typed would silently run a different model
        # family (ADR 0002).
        return "seedvr2"

    for candidate in (model_path, model_arg):
        if not _looks_like_swiftvr_directory(candidate):
            continue
        if _looks_like_seedvr2_directory(candidate):
            raise ValueError(
                f"{candidate!r} holds both a SwiftVR checkpoint ({SWIFTVR_MARKER_FILE}) and a SeedVR2 "
                "checkpoint, so MLX-Gen cannot tell which one you meant. Pass an explicit handle: "
                "--model swiftvr, or --model seedvr2-3b / seedvr2-7b / seedvr2-7b-sharp."
            )
        return "swiftvr"

    return "seedvr2"


def resolve_restore_family(model_arg: str | None, model_path: str | None) -> RestoreFamilyRoute:
    """Classify the handle and resolve it through its family's own resolver.

    SeedVR2 handles go to ``_resolve_seedvr2_model`` untouched, so every alias, official
    repo id, AbstractFramework package and local-directory probe behaves exactly as it
    did before SwiftVR existed. Its unknown-handle ``ValueError`` is re-raised with the
    SwiftVR handles appended after the original sentence, leaving that sentence intact.

    Raises:
        ValueError: If neither family recognises the handle.
    """
    family = classify_restore_family(model_arg, model_path)
    if family == "swiftvr":
        from mflux.models.swiftvr.cli.swiftvr_restore import resolve_swiftvr_model

        model_config, resolved_path = resolve_swiftvr_model(model_arg, model_path)
        return RestoreFamilyRoute(family="swiftvr", model_config=model_config, model_path=resolved_path)

    from mflux.models.seedvr2.cli.seedvr2_upscale import _resolve_seedvr2_model

    try:
        model_config, resolved_path = _resolve_seedvr2_model(model_arg, model_path)
    except ValueError as exc:
        raise ValueError(f"{exc} For one-step video restoration, use {SWIFTVR_HANDLE_HELP}.") from exc
    return RestoreFamilyRoute(family="seedvr2", model_config=model_config, model_path=resolved_path)


class _PeekParseError(Exception):
    """Raised instead of exiting when the pre-parse peek cannot read ``argv``."""


class _PeekParser(argparse.ArgumentParser):
    """``--model`` extractor that reports failures instead of exiting the process."""

    def error(self, message: str):  # noqa: D102 - argparse override, not new API
        raise _PeekParseError(message)


def _peek_model_options(argv: list[str]) -> tuple[str | None, str | None]:
    """Read ``--model`` and ``--path`` out of raw ``argv`` exactly as argparse would.

    Hand-rolled scanning gets the attached short form wrong: argparse reads
    ``-mswiftvr`` as ``--model swiftvr`` while a token-equality scan sees an unknown
    flag, and the two classifiers then disagree - the command dispatches to SeedVR2 and
    the run dies telling the user to use ``mlxgen upscale``, which is what they typed.
    Delegating to argparse removes that whole class of divergence: the attached form,
    ``--model=x`` and long-option abbreviations all resolve the way the real parser
    resolves them.

    Returns:
        ``(model, path)``, either of which may be ``None``. Both are ``None`` when the
        arguments cannot be parsed at all - the family's own parser then raises the real
        error, which is strictly better than this one guessing at a usage message.
    """
    parser = _PeekParser(add_help=False)
    parser.add_argument("--model", "-m", dest="model", default=None)
    parser.add_argument("--path", dest="path", default=None)
    try:
        known, _unknown = parser.parse_known_args(argv)
    except _PeekParseError:
        return None, None
    return known.model, known.path


def peek_restore_family(argv: list[str]) -> RestoreFamily:
    """Classify from raw ``argv``, before argparse runs.

    Uses the same rules as :func:`classify_restore_family` so the pre-parse and
    post-parse decisions cannot diverge into a run that dispatches to one family's
    ``main()`` and then loads the other family's checkpoint.
    """
    model_arg, model_path = _peek_model_options(argv)
    return classify_restore_family(model_arg, model_path)


def upscale_main() -> None:
    """Entry point for ``mlxgen upscale``: pick the family, then run its ``main()``."""
    import sys

    family = peek_restore_family(sys.argv[1:])
    if family == "swiftvr":
        from mflux.models.swiftvr.cli.swiftvr_restore import main as family_main
    else:
        from mflux.models.seedvr2.cli.seedvr2_upscale import main as family_main
    family_main()
