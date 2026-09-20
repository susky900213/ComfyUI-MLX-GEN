"""按「真实节点定义」校验 workflows/ 下的 MiniMax-H3 工作流（不加载大权重）。

对每份 JSON 检查：

1. 每个节点的 `type` 都能找到定义：本插件的节点直接读根目录 `__init__.py`
   的 `NODE_CLASS_MAPPINGS`（ComfyUI 就是这么注册的），ComfyUI 自带节点
   从安装的源码里扫 `NODE_CLASS_MAPPINGS` 确认存在（不 import，避免把
   torch / comfy_aimdo 等运行时依赖全拖进来）；
2. 本插件节点的输入槽名、槽类型与 `INPUT_TYPES` 一致，`widgets_values`
   的个数与顺序必须等于「定义里的 widget 输入」（否则前端按位置填错 widget）；
   ComfyUI 自带节点只查存在性（`CreateVideo` 的 bit_depth/codec、
   `SaveVideo` 的 format、`ConcatenateVideo` 的 autogrow 都是
   dynamic/autogrow widget，序列化形态跟 `define_schema` 对不上，逐个比会误报）；
3. 视觉条件 ↔ transformer 档位是否配对（参考生视频必须落在 REF 权重上，
   2 张以上参考图同理），并确认声明的权重目录 / 文件确实在磁盘上。

运行：python tools/check_workflows_against_comfyui.py [workflows/xxx.json ...]
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path("/Users/apple/ComfyUI-Installs/ComfyUI/ComfyUI")
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "COMBO", "BOOLEAN", "SECRET", "PYSEED"}
VISUAL_NODES = {"MlxH3KeyframeCondition", "MlxH3MultiReferenceCondition", "MlxH3VideoCondition"}

# ComfyUI 前端对名叫 `seed` / `noise_seed` 的 widget 会自动插一个「控制方式」伴随 widget
# （前端源码里的 `(t === "seed" || t === "noise_seed") && (i.control_after_generate = ...)`，
# 与节点定义里有没有声明这个选项无关），所以工作流 JSON 的 widgets_values 里 seed 后面
# 必须紧跟它的值；漏写会让后面**所有** widget 错位一格（提交时报 `scheduler: 124` /
# `steps: 640` 这类看起来毫无道理的校验错误，而且往往在导入后第一次排队才炸出来）。
CONTROL_AFTER_GENERATE = "control_after_generate"
VALUE_CONTROL_OPTIONS = ["fixed", "increment", "decrement", "randomize"]
SEED_WIDGET_NAMES = {"seed", "noise_seed"}

sys.path.insert(0, str(ROOT / "src"))

from comfyui_mlx_gen import paths as ml_paths  # noqa: E402
from comfyui_mlx_gen.h3 import pipeline as h3_pipeline  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, label: str) -> None:
    print(f"  {'OK   ' if condition else 'FAIL '}{label}")
    if not condition:
        FAILURES.append(label)


# --------------------------------------------------------------------- 节点定义来源
def plugin_node_classes() -> dict[str, object]:
    """读根目录 `__init__.py`（ComfyUI 实际加载的入口）拿本插件节点类。"""
    spec = importlib.util.spec_from_file_location("mlxgen_entry", ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.NODE_CLASS_MAPPINGS)


def core_node_types() -> set[str]:
    """扫描安装的 ComfyUI 源码收集节点名（不 import）：`NODE_CLASS_MAPPINGS`
    的键，以及 v2 节点 `define_schema` 里的 `node_id="..."`。"""
    found: set[str] = set()
    files = [COMFY / "nodes.py", *sorted((COMFY / "comfy_extras").glob("nodes_*.py"))]
    for path in files:
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign)
                    and any(getattr(target, "id", "") == "NODE_CLASS_MAPPINGS"
                            for target in node.targets)
                    and isinstance(node.value, ast.Dict)):
                found.update(key.value for key in node.value.keys
                             if isinstance(key, ast.Constant) and isinstance(key.value, str))
            if isinstance(node, ast.keyword) and node.arg == "node_id":
                value = getattr(node.value, "value", None)
                if isinstance(value, str):
                    found.add(value)
    return found


def normalize_input(name: str, value) -> tuple[str, str, dict]:
    """把 INPUT_TYPES 的一项规范成 (槽名, 类型, {options/default})。"""
    pair = value if isinstance(value, (list, tuple)) and value else [None, {}]
    first = pair[0]
    extra = dict(pair[1]) if len(pair) > 1 and isinstance(pair[1], dict) else {}
    if isinstance(first, (list, tuple)):
        return name, "COMBO", {**extra, "options": list(first)}
    if isinstance(first, str):
        return name, first, extra
    if first is not None:  # v2 API：io.Combo.Input / io.Image.Input 之类
        ntype = getattr(first, "type", None) or getattr(first, "input_type", None)
        options = getattr(first, "options", None)
        default = getattr(first, "default", None)
        if isinstance(options, (list, tuple, set)) and options:
            extra = {**extra, "options": list(options)}
        if default is not None:
            extra = {**extra, "default": default}
        return name, str(ntype) if ntype else "?", extra
    return name, "?", extra


# ----------------------------------------------------------------- 定义 → 校验
def definition_inputs(cls) -> list[tuple[str, str, dict]]:
    """节点定义的输入顺序 → [(槽名, 类型, 额外信息)]。"""
    if hasattr(cls, "define_schema"):
        return [
            normalize_input(getattr(item, "name", None) or getattr(item, "id", None) or "?",
                            [item, {}])
            for item in cls.define_schema().inputs
        ]
    spec = cls.INPUT_TYPES()
    out: list[tuple[str, str, dict]] = []
    for section in ("required", "optional"):
        for name, value in (spec.get(section) or {}).items():
            out.append(normalize_input(name, value))
    return out


def definition_outputs(cls) -> list[tuple[str, str]]:
    """节点定义的输出顺序 → [(槽名, 类型)]。"""
    if hasattr(cls, "define_schema"):
        return [
            (
                getattr(item, "name", None) or getattr(item, "id", None) or "?",
                str(getattr(item, "type", None) or getattr(item, "input_type", None) or "?"),
            )
            for item in cls.define_schema().outputs
        ]
    types = tuple(getattr(cls, "RETURN_TYPES", ()))
    names = getattr(cls, "RETURN_NAMES", None)
    if names and len(names) == len(types):
        return list(zip(names, types))
    return [(str(t), str(t)) for t in types]


def widget_value_ok(ntype: str, extra: dict, value) -> bool:
    options = extra.get("options")
    if isinstance(options, (list, tuple)):
        return value in options
    if ntype in {"INT", "FLOAT"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if ntype == "STRING":
        return isinstance(value, str)
    if ntype == "BOOLEAN":
        return isinstance(value, bool)
    return ntype in {"SECRET", "PYSEED"}  # 其余都是 socket，不该出现在 widget 里


def frontend_widgets(widget_def: list[tuple[str, str, dict]]) -> list[tuple[str, str, dict]]:
    """把「节点定义里的 widget 列表」补成「前端真正渲染出来的 widget 列表」。

    唯一的差别是 seed / noise_seed 后面那个 `control_after_generate` 伴随 widget：
    前端无条件插入它，所以工作流 JSON 的 widgets_values 也必须为它留一个值。
    """
    out: list[tuple[str, str, dict]] = []
    for name, ntype, extra in widget_def:
        out.append((name, ntype, extra))
        if name in SEED_WIDGET_NAMES:
            out.append(
                (CONTROL_AFTER_GENERATE, "COMBO", {"options": list(VALUE_CONTROL_OPTIONS)})
            )
    return out


def check_node(node, inputs, outputs) -> list[str]:
    """对照节点定义检查槽名 / 类型 / widget 数量与取值（返回错误列表）。"""
    errors: list[str] = []
    declared_inputs = node.get("inputs", [])
    declared_names = [item["name"] for item in declared_inputs]
    def_names = [name for name, _, _ in inputs]

    unknown = sorted(set(declared_names) - set(def_names))
    if unknown:
        errors.append(f"节点 {node['id']} ({node['type']}) 有定义里不存在的输入槽：{unknown}")
    known = [name for name in def_names if name in declared_names]
    if known != declared_names:
        errors.append(
            f"节点 {node['id']} ({node['type']}) 输入槽顺序与定义不一致："
            f"{declared_names} vs {def_names}"
        )

    widget_def = frontend_widgets(
        [(name, ntype, extra) for name, ntype, extra in inputs if ntype in WIDGET_TYPES]
    )
    values = node.get("widgets_values", [])
    if len(values) > len(widget_def):
        errors.append(
            f"节点 {node['id']} ({node['type']}) 多写了 "
            f"{len(values) - len(widget_def)} 个 widget 值"
        )
    elif len(values) < len(widget_def):
        tail = [name for name, _, _ in widget_def][len(values):]
        print(f"  [info] 节点 {node['id']} ({node['type']}) 只序列化了 {len(values)} 个 "
              f"widget（定义 {len(widget_def)} 个，尾部 {tail} 走默认值）")
    for index, ((name, ntype, extra), value) in enumerate(zip(widget_def, values)):
        if name == CONTROL_AFTER_GENERATE:
            # 这一格必须写前端给 seed 插的那个伴随 widget（值不是控制方式就是错位了）
            if value not in VALUE_CONTROL_OPTIONS:
                errors.append(
                    f"节点 {node['id']} ({node['type']}) 的 seed 后面少了 "
                    f"control_after_generate 的值（第 {index + 1} 个 widget 位上是 {value!r}）："
                    "ComfyUI 前端会给 seed / noise_seed 自动插这个伴随 widget，漏写会让 "
                    "seed 之后的所有 widget 错位一格（提交时报 scheduler / steps 之类的怪错）；"
                    f"请在 seed 值后面补一个 {'/'.join(VALUE_CONTROL_OPTIONS)}"
                )
            continue
        if not widget_value_ok(ntype, extra, value):
            errors.append(
                f"节点 {node['id']} ({node['type']}) 的 {name}={value!r} 与定义不搭"
                f"（{ntype}, options={extra.get('options')}）"
            )
    for item, (name, ntype, _extra) in zip(declared_inputs, inputs):
        if item["name"] == name and item["type"] != ntype:
            errors.append(
                f"节点 {node['id']} ({node['type']}) 槽 {name} 类型 {item['type']} "
                f"与定义 {ntype} 不一致"
            )

    declared_outputs = node.get("outputs", [])
    if len(declared_outputs) != len(outputs):
        errors.append(
            f"节点 {node['id']} ({node['type']}) 输出数量不符："
            f"序列化 {len(declared_outputs)} 个 / 定义 {len(outputs)} 个"
        )
    for item, (_name, ntype) in zip(declared_outputs, outputs):
        if item["type"] != ntype:
            errors.append(
                f"节点 {node['id']} ({node['type']}) 输出 {item['name']} 类型 "
                f"{item['type']} 与定义 {ntype} 不一致"
            )
    return errors


# ----------------------------------------------------------------- 权重档位校验
def widget_map(node, cls) -> dict:
    """按定义顺序把 widgets_values 对回「槽名 → 值」。"""
    names = [name for name, ntype, _extra in definition_inputs(cls) if ntype in WIDGET_TYPES]
    return dict(zip(names, node.get("widgets_values", [])))


def condition_spec(workflow, plugin_classes) -> SimpleNamespace | None:
    """按工作流声明的视觉条件节点，还原 (图片张数, 锚点) 供档位校验。"""
    nodes_by_id = {item["id"]: item for item in workflow["nodes"]}
    visual = next((item for item in workflow["nodes"] if item["type"] in VISUAL_NODES), None)
    if visual is None:
        return None
    widgets = widget_map(visual, plugin_classes[visual["type"]])
    slots = {item["name"]: item.get("link") for item in visual.get("inputs", [])}

    if visual["type"] == "MlxH3KeyframeCondition":
        anchors = tuple(a for a, slot in (("first", "first_frame"), ("last", "last_frame"))
                        if slots.get(slot) is not None)
        return SimpleNamespace(anchors=anchors, picture_count=len(anchors),
                               source_label="首尾帧")
    if visual["type"] == "MlxH3VideoCondition":
        mode = widgets.get("mode", "continue_from_end")
        return SimpleNamespace(
            anchors=("last",) if mode == "continue_from_end" else ("first",),
            picture_count=1,
            source_label="源视频续写",
        )

    source_id = next((link[1] for link in workflow["links"]
                      if link[3] == visual["id"] and link[4] == 0), None)
    if source_id is None:
        return SimpleNamespace(anchors=(), picture_count=0, source_label="多图参考（未接图集）")
    source = nodes_by_id[source_id]
    count = sum(1 for item in source.get("inputs", []) if item.get("link") is not None)
    anchor = widgets.get("anchor", "none")
    anchors = {"none": (), "first": ("first",), "last": ("last",),
               "first_last": ("first", "last")}.get(anchor, ())
    return SimpleNamespace(anchors=anchors, picture_count=count,
                           source_label=f"{count} 张参考图（{anchor}）")


def check_workflow(path: Path, plugin_classes, core_types) -> None:
    print(f"\n== {path.relative_to(ROOT)}")
    workflow = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    existence_only: list[str] = []
    for node in workflow["nodes"]:
        ntype = node["type"]
        if ntype in plugin_classes:
            cls = plugin_classes[ntype]
            errors += check_node(node, definition_inputs(cls), definition_outputs(cls))
        elif ntype in core_types:
            existence_only.append(ntype)
        else:
            errors.append(f"节点 {node['id']} ({ntype}) 在插件与 ComfyUI 里都找不到定义")
    check(not errors, "所有节点的槽位 / widget 与节点定义一致"
          + ("" if not errors else "；问题：" + " | ".join(errors[:8])))
    if existence_only:
        print(f"  [info] 只查存在性的 ComfyUI 自带节点：{', '.join(sorted(set(existence_only)))}")

    visual_spec = condition_spec(workflow, plugin_classes)
    loader = next(item for item in workflow["nodes"] if item["type"] == "MlxTransformerLoader")
    # 「MLX 模型加载」里这个 widget 就叫 model_path（目录名 / 单文件名，相对 models/mlx）
    model_path = str(widget_map(loader, plugin_classes["MlxTransformerLoader"]).get("model_path", ""))
    exists = (ml_paths.component_dir("transformer") / model_path).exists()
    if visual_spec is None:
        check(exists, f"无视觉条件，transformer {model_path!r} 存在={exists}")
        return
    try:
        hint = h3_pipeline.check_visual_condition_checkpoint(
            SimpleNamespace(model_path=model_path), visual_spec
        )
    except ValueError as exc:
        check(False, f"条件 {visual_spec.source_label} + {model_path} 判定不可用：{exc}")
        return
    expected = h3_pipeline.checkpoint_tier_for_condition(visual_spec)
    is_ref = h3_pipeline.uses_ref_checkpoint(model_path)
    check(
        exists and ((expected == "ref") == is_ref or expected == "any"),
        f"条件 {visual_spec.source_label} → 应走 {expected} 档，实际 {model_path!r}"
        f"（存在={exists}，REF={is_ref}）；{hint or '无提示'}",
    )


def main() -> None:
    plugin_classes = plugin_node_classes()
    core_types = core_node_types()
    targets = [Path(arg) for arg in sys.argv[1:]] or sorted(
        (ROOT / "workflows").glob("minimax-h3-*.json")
    )
    for path in targets:
        check_workflow(path, plugin_classes, core_types)
    if FAILURES:
        print(f"\n{len(FAILURES)} 项失败：{' | '.join(FAILURES)}")
        raise SystemExit(1)
    print("\n全部工作流与节点定义对得上。")


if __name__ == "__main__":
    main()
