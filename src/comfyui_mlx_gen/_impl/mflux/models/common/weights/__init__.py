from typing import TYPE_CHECKING

__all__ = [
    "ComponentDefinition",
    "LoadedWeights",
    "MetaData",
    "ModelSaver",
    "WeightApplier",
    "WeightLoader",
]

if TYPE_CHECKING:
    from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.common.weights.saving.model_saver import ModelSaver


def __getattr__(name: str):
    if name in {"LoadedWeights", "MetaData"}:
        from mflux.models.common.weights.loading import loaded_weights

        return getattr(loaded_weights, name)
    if name == "WeightApplier":
        from mflux.models.common.weights.loading.weight_applier import WeightApplier

        return WeightApplier
    if name == "ComponentDefinition":
        from mflux.models.common.weights.loading.weight_definition import ComponentDefinition

        return ComponentDefinition
    if name == "WeightLoader":
        from mflux.models.common.weights.loading.weight_loader import WeightLoader

        return WeightLoader
    if name == "ModelSaver":
        from mflux.models.common.weights.saving.model_saver import ModelSaver

        return ModelSaver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
