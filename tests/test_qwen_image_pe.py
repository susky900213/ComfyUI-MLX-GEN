"""Qwen-Image 2.1 PE 的无权重回归测试。

覆盖消息格式、输出清理、路径发现、懒依赖、节点签名以及 fake
Transformers / MLX-VLM bundle 的缓存 / 生成流程；不会下载或加载真实大模型。
"""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comfyui_mlx_gen import NODE_CLASS_MAPPINGS  # noqa: E402
from comfyui_mlx_gen import paths  # noqa: E402
pe = importlib.import_module("comfyui_mlx_gen.qwen_image_pe")  # noqa: E402
from comfyui_mlx_gen.cache import Cache  # noqa: E402
from comfyui_mlx_gen.nodes.text_encoder import MlxTextEncoder  # noqa: E402
from comfyui_mlx_gen.nodes.qwen_image_pe import _rewritten_prompt_result  # noqa: E402
from comfyui_mlx_gen.types import MlxClipHandle  # noqa: E402


FAILED: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK ' if ok else 'FAIL'}] {label}" + (f" | {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


def check_raises(label: str, exc_type: type[BaseException], part: str, call) -> None:
    try:
        call()
    except exc_type as exc:
        check(label, part in str(exc), str(exc))
    except Exception as exc:  # noqa: BLE001
        check(label, False, f"异常类型不对：{type(exc).__name__}: {exc}")
    else:
        check(label, False, f"没有抛出 {exc_type.__name__}")


def make_checkpoint(
    root: Path, name: str, mlx: bool = False, mlx_variant: str | None = "4bit"
) -> Path:
    path = root / "text_encoder" / name
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "system_prompt.txt").write_text("system instructions", encoding="utf-8")
    (path / "README.md").write_text(
        f"# {name}\nbase_model: {name}\npipeline_tag: text-generation\n",
        encoding="utf-8",
    )
    if mlx:
        variant = path / mlx_variant if mlx_variant else path
        variant.mkdir(parents=True, exist_ok=True)
        for name_ in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            (variant / name_).write_text("{}", encoding="utf-8")
        (variant / "model.safetensors").write_bytes(b"fake")
    return path


class FakeBatch(dict):
    def to(self, device):
        self["device"] = device
        return self


class FakeTokenizer:
    def __init__(self, seen: dict):
        self.seen = seen

    def apply_chat_template(self, messages, **kwargs):
        self.seen["template"] = (messages, kwargs)
        return "formatted prompt"

    def __call__(self, text, **kwargs):
        self.seen["tokenize"] = (text, kwargs)
        return FakeBatch({"input_ids": torch.tensor([[10, 11, 12]])})

    def decode(self, tokens, **kwargs):
        self.seen["decode"] = (tokens, kwargs)
        return '<think>internal</think>\n```json\n{"rewritten_prompt": "a polished prompt"}\n```'


class FakeProcessor:
    def __init__(self, seen: dict):
        self.seen = seen
        self.tokenizer = FakeTokenizer(seen)

    def apply_chat_template(self, messages, **kwargs):
        self.seen["processor_template"] = (messages, kwargs)
        return FakeBatch({
            "input_ids": torch.tensor([[20, 21]]),
            "pixel_values": torch.zeros((1, 3, 2, 2)),
        })


class FakeModel:
    def __init__(self, seen: dict):
        self.seen = seen
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.seen["generation"] = kwargs
        input_ids = kwargs["input_ids"]
        return torch.cat((input_ids, torch.tensor([[30, 31]])), dim=-1)


def run() -> None:
    # ----------------------------------------------------------- 1. 消息与 JSON 清理
    t2i = pe.build_t2i_messages("sys", "a cat")
    check(
        "T2I 使用 system/user 文本消息",
        t2i == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a cat"},
        ],
        repr(t2i),
    )
    image = Image.new("RGBA", (4, 3), (255, 0, 0, 127))
    i2i = pe.build_i2i_messages("sys", [image], "replace sky")
    check(
        "I2I 把图片转 RGB 并将文本放在同一 user content",
        i2i[0] == {"role": "system", "content": [{"type": "text", "text": "sys"}]}
        and i2i[1]["content"][-1] == {"type": "text", "text": "replace sky"}
        and i2i[1]["content"][0]["image"].mode == "RGB",
        repr(i2i),
    )
    check(
        "清理 thinking、JSON fence 与 rewritten_prompt",
        pe.clean_generation(
            '<think>reason</think> {"rewritten_prompt":"  detailed scene  "}<|im_end|>'
        )
        == "detailed scene",
    )
    check_raises(
        "拒绝缺少 rewritten_prompt 的 JSON",
        ValueError,
        "rewritten_prompt",
        lambda: pe.clean_generation('{"prompt":"wrong key"}'),
    )

    # ----------------------------------------------------------- 2. 路径 / 发现与节点注册
    old_root = paths.MODEL_ROOT
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        t2i_path = make_checkpoint(root, pe.T2I_PREFIX)
        make_checkpoint(root, pe.I2I_PREFIX + "-custom")
        nested = t2i_path / "8bit" / (pe.I2I_PREFIX + "-nested")
        nested.mkdir(parents=True)
        (nested / "README.md").write_text(
            f"# {pe.I2I_PREFIX}-nested\n", encoding="utf-8"
        )
        paths.MODEL_ROOT = root
        try:
            check(
                "只发现 text_encoder 直接子目录中的 T2I / I2I PE 模型",
                pe.model_options("t2i") == [pe.T2I_PREFIX]
                and pe.model_options("i2i") == [pe.I2I_PREFIX + "-custom"],
                f"{pe.model_options('t2i')} / {pe.model_options('i2i')}",
            )
            check(
                "Transformers 模型名和绝对路径都能解析并校验必需文件",
                pe.resolve_model_path(pe.T2I_PREFIX, "t2i", pe.TRANSFORMERS) == t2i_path.resolve()
                and pe.resolve_model_path(t2i_path, "t2i", pe.TRANSFORMERS) == t2i_path.resolve(),
            )
            (t2i_path / "system_prompt.txt").unlink()
            check_raises(
                "模型缺 system_prompt.txt 时提前报错",
                FileNotFoundError,
                "system_prompt.txt",
                lambda: pe.resolve_model_path(t2i_path, "t2i"),
            )
            check_raises(
                "拒绝 stale workflow 的 T2I/I2I 权重选择",
                ValueError,
                pe.I2I_PREFIX,
                lambda: pe.resolve_model_path(pe.I2I_PREFIX + "-custom", "t2i", pe.TRANSFORMERS),
            )
        finally:
            paths.MODEL_ROOT = old_root

    t2i_node = NODE_CLASS_MAPPINGS["MlxQwenImagePET2I"]
    i2i_node = NODE_CLASS_MAPPINGS["MlxQwenImagePEI2I"]
    check(
        "T2I / I2I PE 节点已注册且输出 STRING",
        {"MlxQwenImagePET2I", "MlxQwenImagePEI2I"} <= set(NODE_CLASS_MAPPINGS)
        and t2i_node.RETURN_TYPES == ("STRING",)
        and i2i_node.RETURN_TYPES == ("STRING",),
    )
    check(
        "PE 输出同时提供前端显示值且保留 STRING result",
        _rewritten_prompt_result("a generated prompt")
        == {
            "ui": {"rewritten_prompt": ["a generated prompt"]},
            "result": ("a generated prompt",),
        },
    )
    web_extension = ROOT / "web" / "qwen_image_pe_prompt_sync.js"
    web_source = web_extension.read_text(encoding="utf-8")
    check(
        "前端扩展只同步 PE → MlxTextEncoder.prompt 的 text widget",
        '"MlxQwenImagePET2I"' in web_source
        and '"MlxQwenImagePEI2I"' in web_source
        and 'target?.comfyClass !== TEXT_ENCODER' in web_source
        and 'input?.name !== "prompt"' in web_source
        and 'item?.name === "text"' in web_source
        and 'widget.value = rewrittenPrompt' in web_source,
        str(web_extension),
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        make_checkpoint(root, pe.T2I_PREFIX)
        make_checkpoint(root, pe.I2I_PREFIX + "-custom")
        old_root = paths.MODEL_ROOT
        paths.MODEL_ROOT = root
        try:
            t2i_model_input = t2i_node.INPUT_TYPES()["required"]["model_path"]
            i2i_model_input = i2i_node.INPUT_TYPES()["required"]["model_path"]
            check(
                "PE 节点从 text_encoder 扫描模型并显示下拉选项",
                t2i_model_input[0] == [pe.T2I_PREFIX]
                and t2i_model_input[1]["default"] == pe.T2I_PREFIX
                and i2i_model_input[0] == [pe.I2I_PREFIX + "-custom"]
                and i2i_model_input[1]["default"] == pe.I2I_PREFIX + "-custom"
                and t2i_node.INPUT_TYPES()["required"]["loader_type"][0] == list(pe.LOADER_OPTIONS)
                and t2i_node.INPUT_TYPES()["required"]["loader_type"][1]["default"] == pe.MLX_VLM_4BIT,
                repr((t2i_model_input, i2i_model_input)),
            )
        finally:
            paths.MODEL_ROOT = old_root
    text_inputs = MlxTextEncoder.INPUT_TYPES()
    check(
        "文本编码器保留旧位置参数并增加 forceInput prompt socket",
        list(text_inputs["optional"]) == ["h3_keyframes", "ref_images", "prompt"]
        and text_inputs["optional"]["prompt"][1]["forceInput"] is True,
    )
    audio_clip = MlxClipHandle(
        model_type="breeze_tts2",
        component="text_encoder",
        source="local",
        path="Breeze-TTS-2-test",
        precision="bfloat16",
        max_length=512,
        cache_key="clip:pe-override",
    )
    positional_condition = MlxTextEncoder().encode(
        "legacy widget text", audio_clip, None, None, "rewritten external prompt"
    )[0]
    check(
        "外部 prompt 覆盖 widget 且旧的 h3/ref 位置参数仍兼容",
        positional_condition.text == "rewritten external prompt",
        positional_condition.text,
    )

    # ----------------------------------------------------------- 3. fake Transformers 生成 / cache
    old_cache = pe.CACHE
    old_loader = pe.QwenImagePEBackend._load_bundle
    old_runtime_key = pe.runtime.cache_key
    old_optional = pe._torch_and_transformers
    seen: dict = {}
    load_count = {"t2i": 0, "i2i": 0}

    def fake_loader(self, kind, path, loader_type=pe.TRANSFORMERS):
        load_count[kind] += 1
        if kind == "t2i":
            return pe._Bundle(
                kind, path, FakeModel(seen), tokenizer=FakeTokenizer(seen), loader_type="transformers"
            )
        return pe._Bundle(
            kind, path, FakeModel(seen), processor=FakeProcessor(seen), loader_type="transformers"
        )

    pe.CACHE = Cache()
    pe.QwenImagePEBackend._load_bundle = fake_loader
    pe._torch_and_transformers = lambda: (torch, None, None, None, None)
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            t2i_path = make_checkpoint(root, pe.T2I_PREFIX + "-a")
            i2i_path = make_checkpoint(root, pe.I2I_PREFIX + "-a")
            backend = pe.QwenImagePEBackend()
            first = backend.enhance_t2i("cat", t2i_path)
            second = backend.enhance_t2i("cat again", t2i_path)
            check(
                "T2I 生成解码并复用同一 cache bundle",
                first == second == "a polished prompt" and load_count["t2i"] == 1,
                f"{first!r}, loads={load_count}",
            )
            i2i_result = backend.enhance_i2i([image], "edit", i2i_path)
            messages, template_kwargs = seen["processor_template"]
            check(
                "I2I 走 processor、多模态消息并与 T2I 分开缓存",
                i2i_result == "a polished prompt"
                and load_count == {"t2i": 1, "i2i": 1}
                and messages[1]["content"][0]["image"].mode == "RGB"
                and template_kwargs["return_dict"] is True,
                repr(seen.get("processor_template")),
            )
            check(
                "生成只解码新 token 并传递采样参数",
                seen["decode"][0].numel() == 2
                and seen["generation"]["do_sample"] is True
                and seen["generation"]["top_k"] == 20,
                repr(seen.get("generation")),
            )

            # PE bucket cap=2：第三个模型应释放最早的 bundle。
            extra = make_checkpoint(root, pe.T2I_PREFIX + "-b")
            third = make_checkpoint(root, pe.T2I_PREFIX + "-c")
            backend.enhance_t2i("b", extra)
            oldest_key = backend._cache_key("t2i", t2i_path)
            oldest_bundle, hit = pe.CACHE.get("qwen_image_pe_transformers_t2i_module", oldest_key)
            check("PE cache 容量为 2 并驱逐最早模型", not hit, repr(oldest_bundle))
            backend.enhance_t2i("c", third)
    finally:
        pe.CACHE = old_cache
        pe.QwenImagePEBackend._load_bundle = old_loader
        pe._torch_and_transformers = old_optional
        pe.runtime.cache_key = old_runtime_key

    # ----------------------------------------------------------- 4. fake MLX-VLM 生成 / MLX 布局
    old_mlx = pe._mlx_vlm
    mlx_seen: dict = {}

    class FakeMlxResult:
        text = '<think>internal</think>{"rewritten_prompt":"mlx prompt"}'

    def fake_apply(processor, config, prompt, **kwargs):
        mlx_seen["template"] = (processor, config, prompt, kwargs)
        return "formatted mlx prompt"

    def fake_generate(model, processor, prompt, **kwargs):
        mlx_seen["generation"] = (model, processor, prompt, kwargs)
        return FakeMlxResult()

    def fake_mlx():
        return fake_apply, fake_generate, lambda path, **kwargs: ("model", "processor"), lambda path: {
            "model_type": "qwen3_5"
        }

    pe._mlx_vlm = fake_mlx
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mlx_t2i = make_checkpoint(root, pe.T2I_PREFIX + "-MLX", mlx=True)
            mlx_i2i = make_checkpoint(root, pe.I2I_PREFIX + "-MLX", mlx=True)
            old_root = paths.MODEL_ROOT
            paths.MODEL_ROOT = root
            try:
                resolved = pe.resolve_model_path(mlx_t2i, "t2i", pe.MLX_VLM_4BIT)
                check(
                    "MLX-VLM 识别 4bit checkpoint 且不要求 system_prompt.txt",
                    resolved == (mlx_t2i / "4bit").resolve(),
                    str(resolved),
                )
                (mlx_t2i / "4bit" / "system_prompt.txt").unlink(missing_ok=True)
                (mlx_t2i / "system_prompt.txt").unlink(missing_ok=True)
                backend = pe.QwenImagePEBackend()
                check(
                    "MLX-VLM T2I 使用 chat template 并清理 JSON",
                    backend.enhance_t2i_with_loader("cat", mlx_t2i, pe.MLX_VLM_4BIT)
                    == "mlx prompt"
                    and mlx_seen["generation"][3]["image"] is None,
                    repr(mlx_seen),
                )
                check(
                    "MLX-VLM I2I 传递 PIL 图片",
                    backend.enhance_i2i_with_loader(
                        [image], "edit", mlx_i2i, pe.MLX_VLM_4BIT
                    )
                    == "mlx prompt"
                    and len(mlx_seen["generation"][3]["image"]) == 1
                    and mlx_seen["generation"][3]["image"][0].mode == "RGB",
                    repr(mlx_seen),
                )
                check(
                    "MLX-VLM 为思考型 PE 提供足够的输出 token",
                    mlx_seen["generation"][3]["max_tokens"] == 8192,
                    repr(mlx_seen["generation"]),
                )
                mlx_16bit = make_checkpoint(
                    root, pe.T2I_PREFIX + "-MLX-16bit", mlx=True, mlx_variant=None
                )
                resolved_16bit = pe.resolve_model_path(mlx_16bit, "t2i", pe.MLX_16BIT)
                check(
                    "MLX 16bit 选项映射到 mlx-vlm 并使用根目录非量化权重",
                    pe.MLX_16BIT in pe.LOADER_OPTIONS
                    and pe.normalize_loader_type(pe.MLX_16BIT) == ("mlx_vlm", "16bit")
                    and pe.normalize_loader_type("fp16") == ("mlx_vlm", "16bit")
                    and resolved_16bit == mlx_16bit.resolve(),
                    str(resolved_16bit),
                )
                mlx_bf16 = make_checkpoint(
                    root, pe.T2I_PREFIX + "-MLX-bf16", mlx=True, mlx_variant="bf16"
                )
                check(
                    "MLX 16bit 兼容 bf16 子目录命名",
                    pe.resolve_model_path(mlx_bf16, "t2i", pe.MLX_16BIT)
                    == (mlx_bf16 / "bf16").resolve(),
                    str(pe.resolve_model_path(mlx_bf16, "t2i", pe.MLX_16BIT)),
                )
            finally:
                paths.MODEL_ROOT = old_root
    finally:
        pe._mlx_vlm = old_mlx

    old_torch = pe._torch_and_transformers
    pe._torch_and_transformers = lambda: (_ for _ in ()).throw(
        RuntimeError("optional dependency missing")
    )
    try:
        check_raises(
            "可选依赖错误在执行时才抛出",
            RuntimeError,
            "optional dependency",
            pe.ensure_optional_dependencies,
        )
    finally:
        pe._torch_and_transformers = old_torch

    if FAILED:
        print(f"\n{len(FAILED)} 项检查失败：")
        for label in FAILED:
            print(f"  - {label}")
        raise SystemExit(1)
    print("\n全部通过：Qwen-Image 2.1 PE 无权重回归测试有效。")


if __name__ == "__main__":
    run()