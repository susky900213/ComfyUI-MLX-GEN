from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any


class WanTextEncoderLoader:
    BERNINI_PRECISION_POLICY_ID = "bernini-umt5-official-v2"

    _load_lock = RLock()
    _missing = object()

    @classmethod
    def load(cls, *, text_encoder_path: Path, torch_dtype: Any, bernini_compatibility: bool) -> Any:
        from transformers import UMT5EncoderModel

        with cls._precision_scope(UMT5EncoderModel, bernini_compatibility=bernini_compatibility):
            return UMT5EncoderModel.from_pretrained(
                text_encoder_path,
                torch_dtype=torch_dtype,
                local_files_only=True,
            )

    @classmethod
    def precision_policy_id(cls, model_config) -> str | None:
        if model_config is not None and bool(
            model_config.transformer_overrides.get("supports_bernini_renderer", False)
        ):
            return cls.BERNINI_PRECISION_POLICY_ID
        return None

    @classmethod
    @contextmanager
    def _precision_scope(cls, model_class, *, bernini_compatibility: bool) -> Iterator[None]:
        with cls._load_lock:
            yield
