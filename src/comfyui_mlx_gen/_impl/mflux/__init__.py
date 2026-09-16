import importlib
import os

# Set TOKENIZERS_PARALLELISM to avoid fork warning
# This must be set before any tokenizers are imported/used
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mflux.python_runtime import (
    GeneratedOutput,
    GenerationRuntimePlan,
    LoadedGenerationModel,
    load_generation_model,
    load_generation_model_for_plan,
    resolve_generation_runtime,
    resolve_generation_runtime_for_plan,
)
from mflux.release.validation_registry import (
    FLUX2_KLEIN_BASE_STARSHIP_PROFILE_ID,
    I2I_EDIT_5X4_PROFILE_ID,
    REFRAME_OUTPAINT_PROFILE_ID,
    ModelValidation,
    ValidationProfile,
    ValidationRecord,
    get_model_validation,
    get_validation_profile,
    list_validation_profiles,
)
from mflux.task_inference import (
    GenerationCapability,
    GenerationPlan,
    ModelCapabilities,
    ResolvedTask,
    RestorationCapability,
    TaskInferenceError,
    get_model_capabilities,
    infer_task,
    normalize_i2i_mode,
    normalize_task,
    resolve_generation_plan,
    resolve_task,
)

# Outpaint entry points are resolved on first use. `mflux.outpaint` reaches PIL through
# OutpaintUtil, and `import mflux` is kept free of PIL (tests/test_import_hygiene.py), so an
# eager import here would put that cost on every host - including the ones that never outpaint.
_LAZY_EXPORTS = {
    name: "mflux.outpaint"
    for name in (
        "OutpaintContract",
        "OutpaintError",
        "OutpaintFillPlan",
        "OutpaintPassPlan",
        "OutpaintRequest",
        "OutpaintSession",
        "ReframeSession",
        "guard_outpaint_fill_plan",
        "outpaint_contract",
        "outpaint_contract_for_model",
        "prepare_outpaint",
        "prepare_reframe",
        "resolve_outpaint_fill_plan",
        "resolve_outpaint_pass_plan",
        "run_outpaint",
    )
}

__all__ = [
    "GenerationCapability",
    "GeneratedOutput",
    "FLUX2_KLEIN_BASE_STARSHIP_PROFILE_ID",
    "GenerationPlan",
    "GenerationRuntimePlan",
    "I2I_EDIT_5X4_PROFILE_ID",
    "LoadedGenerationModel",
    "ModelValidation",
    "ModelCapabilities",
    "OutpaintContract",
    "OutpaintError",
    "OutpaintFillPlan",
    "OutpaintPassPlan",
    "OutpaintRequest",
    "OutpaintSession",
    "REFRAME_OUTPAINT_PROFILE_ID",
    "ReframeSession",
    "ResolvedTask",
    "RestorationCapability",
    "TaskInferenceError",
    "ValidationProfile",
    "ValidationRecord",
    "get_model_capabilities",
    "get_model_validation",
    "get_validation_profile",
    "guard_outpaint_fill_plan",
    "infer_task",
    "list_validation_profiles",
    "load_generation_model",
    "load_generation_model_for_plan",
    "normalize_i2i_mode",
    "normalize_task",
    "outpaint_contract",
    "outpaint_contract_for_model",
    "prepare_outpaint",
    "prepare_reframe",
    "resolve_generation_plan",
    "resolve_generation_runtime",
    "resolve_generation_runtime_for_plan",
    "resolve_outpaint_fill_plan",
    "resolve_outpaint_pass_plan",
    "resolve_task",
    "run_outpaint",
]


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
