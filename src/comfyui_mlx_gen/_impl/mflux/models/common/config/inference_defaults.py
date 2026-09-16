MODEL_INFERENCE_STEPS = {
    "dev": 25,
    "schnell": 4,
    "krea-dev": 25,
    "qwen": 20,
    "qwen-image": 20,
    "qwen-image-edit": 20,
    "qwen-image-edit-2509": 40,
    "qwen-image-edit-2511": 40,
    "qwen-edit": 20,
    "qwen-edit-plus": 40,
    "qwen-edit-2509": 40,
    "qwen-edit-2511": 40,
    "fibo": 50,
    "fibo-lite": 8,
    "fibo-edit": 50,
    "fibo-edit-rmbg": 10,
    "z-image": 50,
    "z-image-turbo": 9,
    "ernie-image-turbo": 8,
    "wan2.2-ti2v-5b": 50,
    "bonsai-image-ternary": 4,
    "bonsai-image-binary": 4,
    "flux2-klein-4b": 4,
    "flux2-klein-9b": 4,
    "flux2-klein-base-4b": 50,
    "flux2-klein-base-9b": 50,
}


def default_inference_steps(
    model_config: object | None = None,
    *,
    model_name: str | None = None,
    base_model: str | None = None,
    fallback: int = 25,
) -> int:
    candidates = [
        model_name,
        *(getattr(model_config, "aliases", ()) if model_config is not None else ()),
        getattr(model_config, "model_name", None) if model_config is not None else None,
        getattr(model_config, "base_model", None) if model_config is not None else None,
        base_model,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        step_count = MODEL_INFERENCE_STEPS.get(str(candidate).lower())
        if step_count is not None:
            return step_count
    return fallback
