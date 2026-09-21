#!/usr/bin/env python3
"""生成 MiniMax-H3 二阶段（低分 → latent 放大 → 高分精修）示例工作流（可重复运行）。

    python tools/gen_h3_two_stage_workflow.py

产出：`workflows/minimax-h3-two-stage-upscale.json`

写法与 `tools/gen_h3_workflows.py` 一致（声明式节点 + 由连线推导 slot/link），
但**自包含**：不去 import 那个脚本（它有模块级副作用，会顺带重写另外 7 份工作流）。
工作流只用到「既有节点 + 本次新增的三个节点」：

    MlxH3FirstPassSampler（种子/步数/画布/帧数/两种 shift/stop_at_step）
      → MlxH3LatentUpscaler（只给倍率，目标画布由上一段推导）
      → MlxH3SecondPassSampler（剩余 σ + 音频三模式）
      → 既有 MlxVaeLoader / MlxVAEDecoder / MlxPilToTorch / CreateVideo / SaveVideo

widgets 的顺序必须与各节点 `INPUT_TYPES` 完全一致，且 seed 后面要紧跟前端自动插入的
`control_after_generate` 值（漏写会让后面所有 widget 错位一格）；
`tools/check_workflows_against_comfyui.py` 会把这些都校验一遍。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workflows"
sys.path.insert(0, str(ROOT / "src"))

from comfyui_mlx_gen import paths  # noqa: E402
from comfyui_mlx_gen.h3 import two_stage  # noqa: E402

PROMPT = (
    "integrated_multimodal_description: 视频共 1 个镜头，清晨湖面上一只白鹭捕鱼："
    "长焦缓慢推近，白鹭立于灰黑礁石，微微俯身探出长喙，把一条银白小鱼甩出水面，"
    "逆光里鱼身闪了一下，白鹭仰头吞下并抖了抖羽毛。自然光、低饱和冷调，"
    "镜头轻推带轻微手持呼吸感。\n\n"
    "overall_soundscape: 清晨户外环境音，轻风、远处零星鸟鸣、江水缓流与水花轻拍礁石；"
    "捕鱼瞬间有短促水花与翅膀扑动声。\n\n"
    "non_diegetic_music: 极简钢琴与弦乐，节奏舒缓、音量克制。"
)

CKPT = "MiniMax-H3"


def node_spec(nid, ntype, title, inputs, outputs, widgets, pos, size, order):
    return {
        "id": nid,
        "type": ntype,
        "title": title,
        "inputs": inputs,  # [[槽名, 类型], ...]
        "outputs": outputs,  # [[槽名, 类型], ...]
        "widgets": widgets,
        "pos": pos,
        "size": size,
        "order": order,
    }


def build_workflow(wf_id, title, nodes, links, groups=()):
    """按 spec 拼出 ComfyUI 工作流：连线决定 inputs[].link 与 outputs[].links。"""
    inputs = {
        spec["id"]: [
            {"name": name, "type": ntype, "link": None, "slot_index": index}
            for index, (name, ntype) in enumerate(spec["inputs"])
        ]
        for spec in nodes
    }
    outputs = {
        spec["id"]: [
            {"name": name, "type": ntype, "links": [], "slot_index": index}
            for index, (name, ntype) in enumerate(spec["outputs"])
        ]
        for spec in nodes
    }
    out_links = {spec["id"]: {} for spec in nodes}
    json_links = []
    for link_id, (src, src_slot, dst, dst_slot, ntype) in enumerate(sorted(links), start=1):
        json_links.append([link_id, src, src_slot, dst, dst_slot, ntype])
        inputs[dst][dst_slot]["link"] = link_id
        out_links[src].setdefault(src_slot, []).append(link_id)
    for spec in nodes:
        for slot, ids in out_links[spec["id"]].items():
            outputs[spec["id"]][slot]["links"] = ids
    json_nodes = []
    for spec in sorted(nodes, key=lambda item: item["order"]):
        json_nodes.append(
            {
                "id": spec["id"],
                "type": spec["type"],
                "pos": spec["pos"],
                "size": spec["size"],
                "flags": {},
                "order": spec["order"],
                "mode": 0,
                "inputs": inputs[spec["id"]],
                "outputs": outputs[spec["id"]],
                "properties": {"Node name for S&R": spec["type"]},
                "title": spec["title"],
                "widgets_values": spec["widgets"],
            }
        )
    return {
        "id": wf_id,
        "revision": 0,
        "last_node_id": max(spec["id"] for spec in nodes),
        "last_link_id": len(json_links),
        "nodes": json_nodes,
        "links": json_links,
        "groups": list(groups),
        "config": {},
        "extra": {"ds": {"scale": 1, "offset": [0, 0]}},
        "version": 0.4,
        "extra_workflow": {"title": title},
    }


DEFAULT_UPSCALER = "minimax_h3_latent_upscaler_3d_bf16.safetensors"


def upscaler_name() -> str:
    """选择示例工作流使用的神经 3D latent 放大模型。

    二阶段示例必须使用真实的 MiniMax-H3 3D upscaler；不能因为模型未安装就
    静默切换到内置插值。这样生成工作流时缺少模型会立即暴露，而不是生成一个
    表面可运行、实际没有使用目标放大网络的工作流。
    """
    models = [n for n in paths.list_component_items("upscaler") if n.endswith(".safetensors")]
    if DEFAULT_UPSCALER in models:
        print(f"    使用 H3 3D latent 放大模型：{DEFAULT_UPSCALER}")
        return DEFAULT_UPSCALER
    if models:
        fallback = models[0]
        print(
            f"    ⚠️ 找不到首选 H3 3D 放大模型 {DEFAULT_UPSCALER}，"
            f"使用已安装的神经放大模型：{fallback}"
        )
        return fallback
    raise FileNotFoundError(
        "没有找到 H3 3D latent 放大模型。请把 "
        f"{DEFAULT_UPSCALER} 放到 {paths.component_dir('upscaler')} 后重新生成工作流。"
    )


# 加速 LoRA 的候选（按偏好排序）：8 步档用 fl2v 适配器 —— 它标称是首 / 尾帧（FL2VA）
# 任务，社区里也常拿它给「文生视频」换速度；要任务严格匹配就换成
# `minimax_h3_ref2v_lightx2v_turbo_4step_v0.1_*.safetensors`（REF 权重 + <Picture N> 提示词）
# 或 `minimax_h3_taomate_3step_*.safetensors`（未标任务），并同步改两段步数。
LORA_PREFERRED = (
    "minimax_h3_fl2v_lightx2v_turbo_8step_v1.0_resized_avg_rank_24_bf16.safetensors",
    "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16_int8convrot.safetensors",
)


def pick_lora() -> str:
    """挑一颗真实存在的 8 步 MiniMax-H3 适配器；没有就退回「<无 LoRA>」。"""
    available = set(paths.scan_loras())
    for name in LORA_PREFERRED:
        if name in available:
            return name
    eight = [name for name in sorted(available) if "minimax_h3" in name and "8step" in name]
    if eight:
        return eight[0]
    print(
        "⚠️ lora/ 下没有 8 步的 MiniMax-H3 适配器：LoRA 变体会写成「<无 LoRA>」"
        "（仍可运行，但没有加速），请把适配器放进 models/mlx/lora/ 后重跑本脚本"
    )
    return "<无 LoRA>"



def graph(
    *,
    lora: str | None,
    first_steps: int,
    stop_at_step: int,
    second_steps: int,
    start_at_sigma: float,
):
    """拼出二阶段链路；`lora` 为 None 时是基座版。

    LoRA 挂在「一阶段与二阶段**共用**的那条 model 线」上：两个采样器各自物化
    transformer 时都会应用同一份适配器（依据见 USAGE_ZH §7 与 §4.5）。
    """
    sigma_video = round(float(two_stage.first_pass_axis(12.0, first_steps, stop_at_step)[-1]), 4)
    sigma_audio = round(float(two_stage.first_pass_axis(3.0, first_steps, stop_at_step)[-1]), 4)
    nodes = [
    node_spec(1, "MlxClipLoader", "H3 条件编码器 + tokenizer（q8，编码完即释放）",
              [], [["CLIP", "CLIP"]],
              ["minimax_h3", "text_encoder", CKPT, "bfloat16", 8192, 8],
              [0, -140], [380, 260], 0),
    node_spec(3, "MlxTransformerLoader", f"H3 transformer（{CKPT}，q8；compile 会被 H3 强制关掉）",
              [], [["model", "model"]],
              ["minimax_h3", CKPT, 8, "bfloat16", False, 2],
              [0, 180], [380, 240], 1),
    node_spec(4, "MlxVAELoader", "H3 视频 VAE（role=vae）",
              [], [["vae", "mlx_vae"]],
              ["minimax_h3", CKPT, "bfloat16", 8, "vae"],
              [0, 460], [380, 220], 2),
    node_spec(5, "MlxVAELoader", "H3 音频 VAE（role=audio_vae）",
              [], [["vae", "mlx_vae"]],
              ["minimax_h3", CKPT, "bfloat16", 8, "audio_vae"],
              [0, 720], [380, 220], 3),
    node_spec(2, "MlxTextEncoder", "H3 提示词（三段式：画面 / 音效 / 配乐）",
              [["clip", "CLIP"]], [["condition", "condition"]],
              [PROMPT], [420, 120], [420, 360], 4),
    # widgets 顺序：seed / control_after_generate / steps / width / height /
    #               num_frames / video_shift / audio_shift / stop_at_step / output_mode
        node_spec(6, "MlxH3FirstPassSampler",
                  f"① 一阶段：640×352 / 124 帧 / {first_steps} 步，停在第 {stop_at_step} 步"
                  f"（σ_video {sigma_video:g} / σ_audio {sigma_audio:g}），"
                  f"输出 denoised (x0) 交给放大网络",
              [["model", "model"], ["positive", "condition"], ["negative", "condition"]],
              [["latents", "latents"]],
              [20260920, "randomize", first_steps, 640, 352, 124, 12.0, 3.0, stop_at_step,
               "denoised (x0)"],
              [880, 120], [460, 420], 5),
    # widgets 顺序：model_name / scale / precision（没有长宽！目标画布由上一段推导）
    node_spec(7, "MlxH3LatentUpscaler",
              "② latent 放大：640×352 ×2 → 1280×704（MiniMax-H3 3D latent 放大模型）",
              [["latents", "latents"]], [["latents", "latents"]],
              [upscaler_name(), 2.0, "bfloat16"],
              [1380, 120], [460, 200], 6),
    # widgets 顺序：seed / control_after_generate / steps / start_at_sigma / audio_mode
        node_spec(8, "MlxH3SecondPassSampler",
                  f"③ 二阶段：从 σ={start_at_sigma:g} 接着走 {second_steps} 步"
                  f"（总步数 {stop_at_step + second_steps}；音频跟视频同一 p 刻度收尾）",
              [["model", "model"], ["positive", "condition"], ["negative", "condition"],
               ["latents", "latents"]],
              [["latents", "latents"]],
              [20260921, "randomize", second_steps, start_at_sigma, "follow_video"],
              [1380, 400], [460, 300], 7),
    node_spec(9, "MlxVAEDecoder", "④ 解码（视频 VAE + 音频 VAE 共用同一条 latent）",
              [["vae", "mlx_vae"], ["latents", "latents"], ["audio_vae", "mlx_vae"]],
              [["images", "images"]],
              [-1], [1880, 400], [420, 260], 8),
    node_spec(10, "MlxPilToTorch", "PIL → IMAGE / MASK / AUDIO",
              [["images", "images"]],
              [["images", "IMAGE"], ["masks", "MASK"], ["audio", "AUDIO"]],
              [], [2340, 400], [360, 260], 9),
    node_spec(11, "CreateVideo", "帧 + 音轨 → VIDEO（H3 是 24 fps）",
              [["images", "IMAGE"], ["fps", "FLOAT"], ["audio", "AUDIO"],
               ["bit_depth", "COMBO"], ["color_space", "COMBO"], ["codec", "COMBO"]],
              [["VIDEO", "VIDEO"]],
              [24, 8, "sRGB", "none"], [2740, 400], [360, 300], 10),
    node_spec(12, "SaveVideo",
              f"存成 mp4（video/MiniMax-H3-two-stage{'-lora' if lora else ''}）",
              [["video", "VIDEO"], ["filename_prefix", "STRING"], ["format", "COMBO"]],
              [["VIDEO", "VIDEO"]],
              [f"video/MiniMax-H3-two-stage{'-lora' if lora else ''}", "mp4"],
              [3140, 400], [360, 240], 11),
    ]

    links = [
    (1, 0, 2, 0, "CLIP"),
    (3, 0, 6, 0, "model"),
    (2, 0, 6, 1, "condition"),
    (2, 0, 6, 2, "condition"),
    (6, 0, 7, 0, "latents"),
    (3, 0, 8, 0, "model"),
    (2, 0, 8, 1, "condition"),
    (2, 0, 8, 2, "condition"),
    (7, 0, 8, 3, "latents"),
    (4, 0, 9, 0, "mlx_vae"),
    (8, 0, 9, 1, "latents"),
    (5, 0, 9, 2, "mlx_vae"),
    (9, 0, 10, 0, "images"),
    (10, 0, 11, 0, "IMAGE"),
    (10, 2, 11, 2, "AUDIO"),
    (11, 0, 12, 0, "VIDEO"),
    ]

    groups = [
    {"id": 1, "title": "① 一阶段：低分辨率采样（可停在母网格第 k 步）", "bounding": [840, 60, 560, 620],
     "color": "#3f789e", "font_size": 20, "flags": {}},
    {"id": 2, "title": "② latent 放大（只给倍率）", "bounding": [1340, 60, 540, 240],
     "color": "#8A8A3f", "font_size": 20, "flags": {}},
    {"id": 3, "title": "③ 二阶段：剩余 σ 精修 + 音频收尾", "bounding": [1340, 320, 540, 420],
     "color": "#3f789e", "font_size": 20, "flags": {}},
    ]

    if lora:
        # widgets 顺序：lora / strength（model 是 socket）
        nodes.append(
            node_spec(13, "MlxModelLoraApply",
                      f"加速 LoRA：{lora}（strength 保持 1.0；两段共用同一条 model 线）",
                      [["model", "model"]], [["model", "model"]],
                      [lora, 1.0], [420, 640], [420, 160], 5)
        )
        # order 只影响前端的初始排列：把 LoRA 插到 transformer 之后、采样器之前
        for spec in nodes:
            if spec["id"] != 13 and spec["order"] >= 5:
                spec["order"] += 1
        # 关键：把 model 线从「transformer → 两个采样器」换成「transformer → LoRA → 两个采样器」，
        # 保证两段用的是同一套（加速后的）权重
        links = [
            item
            for item in links
            if not (item[2] in (6, 8) and item[3] == 0 and item[4] == "model")
        ]
        links += [(3, 0, 13, 0, "model"), (13, 0, 6, 0, "model"), (13, 0, 8, 0, "model")]
        groups.append(
            {"id": 4, "title": "加速 LoRA（一阶段 + 二阶段共用）",
             "bounding": [380, 580, 500, 260], "color": "#a1309b", "font_size": 20, "flags": {}}
        )

    return nodes, links, groups


def write(name: str, wf_id: str, title: str, nodes, links, groups) -> None:
    workflow = build_workflow(wf_id, title, nodes, links, groups)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写出 {path.relative_to(ROOT)}（{len(workflow['nodes'])} 个节点，{len(workflow['links'])} 条连线）")


def main() -> None:
    upscaler = upscaler_name()
    print(f"放大模型：{upscaler}")

    # ① 基座版：50 步档能用的 8 步母网格切半（二阶段从 σ=0.7 接着走）
    nodes, links, groups = graph(
        lora=None, first_steps=8, stop_at_step=4, second_steps=4, start_at_sigma=0.7
    )
    write(
        "minimax-h3-two-stage-upscale",
        "mlx-minimax-h3-two-stage-upscale",
        "MiniMax-H3 二阶段：低分采样 → latent 放大 → 高分精修",
        nodes,
        links,
        groups,
    )

    # ② 加速 LoRA 版：8 步适配器 → 总步数必须 = 8，所以 4 + 4，且切点取母网格第 4 点
    #    （σ=0.9231，二阶段子网格与母网格后半段逐点一致 = 完整 8 步轨迹，只是后半在 2× 分辨率上）
    lora = pick_lora()
    print(f"加速 LoRA：{lora}")
    nodes, links, groups = graph(
        lora=lora, first_steps=8, stop_at_step=4, second_steps=4, start_at_sigma=0.9231
    )
    write(
        "minimax-h3-two-stage-upscale-lora",
        "mlx-minimax-h3-two-stage-upscale-lora",
        "MiniMax-H3 二阶段（加速 LoRA 版，8 步档 4+4）：低分采样 → latent 放大 → 高分精修",
        nodes,
        links,
        groups,
    )


if __name__ == "__main__":
    main()

