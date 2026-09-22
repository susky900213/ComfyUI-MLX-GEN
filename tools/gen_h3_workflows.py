"""生成 MiniMax-H3 的九份示例工作流（纯标准库，可重复运行）。

写法说明：工作流里每个节点只声明「输入槽名 + 输出槽名 + widget 值」，
链接用 (源节点, 源槽, 目标节点, 目标槽, 类型) 描述；slot_index、
inputs[].link、outputs[].links 与 links 数组都由本脚本统一推导，
避免手写 JSON 时 link / slot 对不上。

任务与权重必须配对（见 `h3/pipeline.check_visual_condition_checkpoint`）：
首 / 尾帧锚点（I2VA、FL2VA、视频续写）用 Base 的 `MiniMax-H3`；
**参考生视频（Ref2VA：多图 / 纯参考）用 Minimax-H3-REF（`MiniMax-H3-ref`）**，
否则采样器会直接报错。文本编码器 / tokenizer / 两个 VAE 两档共用 `MiniMax-H3`。

运行：python tools/gen_h3_workflows.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workflows"

PROMPT_MULTI = (
    "integrated_multimodal_description: 视频共 1 个镜头，按参考图生成：<Picture 1> 是"
    "要照搬的首帧（海边木栈桥的尽头），<Picture 2> 提供黄昏礁石的场景与色调，"
    "<Picture 3> 提供逆光白帆船这个主体。镜头从 <Picture 1> 的栈桥尽头缓缓推向"
    "海平面，海鸥掠过画面前景，夕阳把木板纹理照成暖橙色；转到 <Picture 2> 的礁石"
    "海岸时潮水漫上来，浪花在逆光里发白，最后 <Picture 3> 的白帆船驶到画面中央。"
    "整体自然光、低饱和暖色调，缓慢推镜带轻微手持呼吸感。\n\n"
    "overall_soundscape: 海浪反复拍岸、海鸥零星叫声、木栈桥被踩到的吱呀声；"
    "后半段潮水漫过礁石的水声渐强。\n\n"
    "non_diegetic_music: 极简木吉他与弦乐铺底，节奏舒缓，音量克制。"
)
PROMPT_REF_ONLY = (
    "integrated_multimodal_description: 视频共 1 个镜头，按 <Picture 1> 到 <Picture 3> 的"
    "顺序取风格：第一张的晨雾湖面、第二张的白鹭、第三张的逆光芦苇。镜头从雾中缓缓横移，"
    "白鹭从芦苇后走出并在浅滩停住，画面始终保持柔和的冷调逆光。\n\n"
    "overall_soundscape: 清晨湖面环境音、远处鸟鸣、芦苇被风吹动的沙沙声。\n\n"
    "non_diegetic_music: 钢琴单音与弦乐长音，安静克制。"
)
PROMPT_VIDEO_KEEP = (
    "integrated_multimodal_description: 接上源视频的最后一帧（已钉成第一帧）："
    "镜头从原来的机位继续往后走，主体沿着步道转向海边，画面里陆续出现防波堤、"
    "白色浮标与远处渔船；光线保持源视频那种低饱和冷调，运镜缓慢、带轻微手持感。\n\n"
    "overall_soundscape: 沿用源视频的环境声（风声、海浪、脚步），"
    "转到海边后浪声渐强，渔船汽笛在远处响一次。\n\n"
    "non_diegetic_music: 弦乐与合成器长音，克制、不喧宾夺主。"
)
PROMPT_VIDEO_DROP = (
    "integrated_multimodal_description: 以源视频的第一帧作为结尾倒推：先拍防波堤上无人"
    "的长椅、海雾逐渐变浓，最后一帧回到源视频开头的那片海面。整体冷灰调、"
    "缓慢横移，几乎没有人声。\n\n"
    "overall_soundscape: 只有风声与远处海潮；中段有一只海鸥叫了两声。\n\n"
    "non_diegetic_music: 无旋律的环境合成器铺底。"
)
PROMPT_VIDEO_REPLACE = (
    "integrated_multimodal_description: 从源视频最后一帧继续：镜头缓慢跟上一只"
    "沿着防波堤奔跑的橘猫，最后停在海平线方向的落日；主体与场景衔接源视频，"
    "但节奏更慢、更抒情。\n\n"
    "overall_soundscape: 猫爪踩在混凝土上的细碎声、风声、远处浪涛；"
    "音轨整体替换成配乐（见工作流的「Load Audio」连线）。\n\n"
    "non_diegetic_music: 温暖的钢琴与弦乐，适合作为一段小纪录片的结尾。"
)

PROMPT_MOTION_TRANSFER = (
    "integrated_multimodal_description: 以 <Video 1> 作为动作与镜头运动参考，"
    "保持参考视频中的人物运动节奏、身体朝向、手臂摆动和镜头轨迹；将主体替换为一名"
    "穿白色运动外套的年轻女性，场景改为黄昏的城市天台。动作要跟随参考视频的时间结构，"
    "但人物外观、服装、环境和色调以文字描述为准，不要复制参考视频中的身份。\n\n"
    "overall_soundscape: 城市天台的风声、远处车流与轻微脚步声，动作变化处保留自然的衣料摩擦声。\n\n"
    "non_diegetic_music: 克制的电子氛围音乐，节奏跟随动作但不要盖过环境声。"
)

PROMPT_MOTION_TRANSFER_WITH_IMAGE = (
    "integrated_multimodal_description: 以 <Video 1> 作为动作与镜头运动参考，"
    "保持参考视频中的动作节奏、身体朝向、手臂摆动和镜头轨迹；以 <Picture 1> 作为"
    "人物外观与服装参考，将主体放在黄昏的城市天台。动作要跟随参考视频的时间结构，"
    "但不要复制参考视频中的身份、脸部和服装；优先保持 <Picture 1> 的人物特征。\n\n"
    "overall_soundscape: 城市天台的风声、远处车流与轻微脚步声，动作变化处保留自然的衣料摩擦声。\n\n"
    "non_diegetic_music: 克制的电子氛围音乐，节奏跟随动作但不要盖过环境声。"
)


# --- 7. 全能参考（4 张参考图 + 加速 LoRA）用的提示词 -------------------------------
# H3 训练时用的是三段带标签的文本（`integrated_multimodal_description` /
# `overall_soundscape` / `non_diegetic_music`），并且靠 `<Picture N>` 引用第 N 张参考图
# （这些标签由「MLX 参考图集 → MLX H3 多图参考条件」自动插进 presentation，
# 所以提示词里必须写对编号，否则模型不知道哪张是哪张）。
PROMPT_ALL_REF = (
    "integrated_multimodal_description: 视频共 1 个镜头，四张参考图各出一半力："
    "<Picture 1> 是白衬衫、黑色高丸子头、圆框金属眼镜的女主角；<Picture 2> 是卡其工装"
    "外套、短寸发型的男主角；<Picture 3> 提供黄昏海边木栈桥的场景与逆光色调；"
    "<Picture 4> 提供 VHS 颗粒、粉蓝渐变与 RGB 错位这套复古质感。开场是 <Picture 3> 的"
    "栈桥尽头，<Picture 1> 的女主倚在栏杆上看海，<Picture 2> 的男主从画面右侧走来、"
    "递过两支冰棍，两人转身并肩看落日；全程保持 <Picture 4> 的低饱和粉蓝色调、"
    "轻微扫描线抖动与偶发的色彩错位，最后收在两人并肩的双人定格。\n\n"
    "overall_soundscape: 海浪反复拍岸、海鸥零星叫声、木栈桥被踩到的吱呀声；"
    "中段有冰棍包装袋的窸窣声与两人压低的轻笑，落日前留一段很长的海风。\n\n"
    "non_diegetic_music: 复古合成器铺底配木吉他扫弦，节奏舒缓，音量克制。"
)

# 参考生视频（Ref2VA）的加速适配器；文件名里的 `ref2v` 就是它对应的任务（配
# `<Picture N>` 那套参考提示词），因此要连 REF 权重（`MiniMax-H3-ref`）+ 4 步。
# 同目录里 `..._fl2v_...` 是首 / 尾帧（FL2VA）用的，`..._taomate_3step_...`
# 文件名只写了步数、没写适用任务，混到 Ref2VA 这条链上任务对不上（别照抄参考工作流）。
LORA_H3_REF_4STEP = "minimax_h3_ref2v_lightx2v_turbo_4step_v0.1_resized_avg_rank_20_bf16.safetensors"


# --- 声明式构造工具 ---------------------------------------------------------------
def node_spec(nid, ntype, title, inputs, outputs, widgets, pos, size, order):
    return {
        "id": nid,
        "type": ntype,
        "title": title,
        "inputs": inputs,  # [[槽名, 类型], ...]
        "outputs": outputs,  # [[槽名, 类型], ...]
        "widgets": widgets,  # 只含 widget（被连线的输入槽不占位）
        "pos": pos,
        "size": size,
        "order": order,
    }


def build_workflow(wf_id, title, nodes, links):
    """按 spec 拼出 ComfyUI 工作流：连线决定 inputs[].link 与 outputs[].links。"""
    spec_by_id = {spec["id"]: spec for spec in nodes}
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
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 1, "offset": [0, 0]}},
        "version": 0.4,
        "extra_workflow": {"title": title},
    }


# --- 公共骨架：加载器 → 文本编码 → 采样 → 解码 → 出片 -------------------------------
# H3 有两套 transformer：Base（t2v / 首尾帧 = `MiniMax-H3`）与 REF
# （Ref2VA，参考生视频 = `MiniMax-H3-ref`）。条件编码器、tokenizer 与两个 VAE
# 只有一份（都叫 `MiniMax-H3`），两档共用，因此只有 transformer 的权重集要换。
CKPT_BASE = "MiniMax-H3"
CKPT_REF = "MiniMax-H3-ref"


def base_nodes(prompt, *, sampler_wh=(640, 352), checkpoint=CKPT_BASE,
               visual_id=None, visual_label="视觉条件",
               steps=50, scheduler="flow_match_euler_discrete"):
    """返回骨架节点（1 CLIP、2 文本、3 transformer、4 音频 VAE、5 采样、6 解码、
    7 PIL→张量、8 CreateVideo、9 SaveVideo）；visual_id 为 None 时是纯文生视频。

    `checkpoint` 只影响 `MlxTransformerLoader` / `MlxVAEDecodeRawPIL` / 输出前缀：
    参考生视频（Ref2VA）传 `CKPT_REF`，其余任务保持 `CKPT_BASE`。

    `steps` / `scheduler` 只写进「MLX 采样器」的 widget：装了加速 LoRA 的
    工作流要按适配器的步数填（否则等于白装），并且 scheduler 必须写
    `minimax_h3`，不然采样阶段会打印「H3 不看 scheduler / guidance」的提示。
    """
    prefix = f"video/{checkpoint}"
    text_inputs = [["clip", "CLIP"]]
    if visual_id is not None:
        text_inputs.append(["h3_keyframes", "mlx_h3_keyframes"])
    text_title = (
        f"H3 提示词 + {visual_label}" if visual_id is not None else "H3 提示词（三段式：画面 / 音效 / 配乐）"
    )
    return [
        node_spec(1, "MlxClipLoader", "H3 条件编码器 + tokenizer（q8，编码完即释放）",
                  [], [["CLIP", "CLIP"]],
                  ["minimax_h3", "text_encoder", CKPT_BASE, "bfloat16", 8192, 8],
                  [0, -120], [380, 260], 0),
        node_spec(3, "MlxTransformerLoader",
                  f"H3 transformer（{checkpoint}，q8；compile 会被 H3 强制关掉）",
                  [], [["model", "model"]],
                  ["minimax_h3", checkpoint, 8, "bfloat16", False, 2],
                  [0, 180], [380, 240], 1),
        node_spec(4, "MlxVAELoader", "H3 音频 VAE（role=audio_vae）",
                  [], [["vae", "mlx_vae"]],
                  ["minimax_h3", CKPT_BASE, "bfloat16", 8, "audio_vae"],
                  [0, 540], [380, 220], 2),
        node_spec(2, "MlxTextEncoder", text_title,
                  text_inputs, [["condition", "condition"]],
                  [prompt], [420, 120], [420, 360], 5),
        node_spec(5, "MlxKSamplerMLX",
                  f"H3 采样（打包序列 + 双整流流；{sampler_wh[0]}×{sampler_wh[1]} / 124 帧 / {steps} 步）",
                  [["model", "model"], ["positive", "condition"], ["negative", "condition"],
                   ["ref_images", "mlx_ref_images"]],
                  [["latents", "latents"]],
                  # seed 后面必须紧跟 control_after_generate 的值：ComfyUI 前端会给
                  # 名叫 seed 的 widget 自动插一个伴随 widget，工作流里漏写这个值
                  # 会让后面所有 widget 错位一格（提交时报 scheduler=124 这类怪错）
                  [20260917, "randomize", steps, sampler_wh[0], sampler_wh[1], 1, 1.0,
                   scheduler, 124, 12.0, 3.0, "auto"],
                  [820, 120], [420, 420], 6),
        node_spec(6, "MlxVAEDecodeRawPIL",
                  "H3 解码（视频 VAE 由 widget 现取；音轨靠接进来的 audio_vae）",
                  [["latents", "latents"], ["audio_vae", "mlx_vae"]],
                  [["images", "images"]],
                  ["minimax_h3", CKPT_BASE, "bfloat16", 8, -1],
                  [820, 600], [420, 320], 7),
        node_spec(7, "MlxPilToTorch", "PIL → IMAGE / MASK / AUDIO",
                  [["images", "images"]],
                  [["images", "IMAGE"], ["masks", "MASK"], ["audio", "AUDIO"]],
                  [], [1300, 600], [360, 260], 8),
        node_spec(8, "CreateVideo", "帧 + 音轨 → VIDEO（H3 是 24 fps）",
                  [["images", "IMAGE"], ["fps", "FLOAT"], ["audio", "AUDIO"],
                   ["bit_depth", "COMBO"], ["color_space", "COMBO"], ["codec", "COMBO"]],
                  [["VIDEO", "VIDEO"]],
                  [24, 8, "sRGB", "none"], [1700, 600], [360, 300], 9),
        node_spec(9, "SaveVideo", f"存成 mp4（{prefix}）",
                  [["video", "VIDEO"], ["filename_prefix", "STRING"], ["format", "COMBO"]],
                  [["VIDEO", "VIDEO"]],
                  [prefix, "mp4"], [2100, 600], [360, 240], 10),
    ]


BASE_LINKS = [
    (1, 0, 2, 0, "CLIP"),
    (3, 0, 5, 0, "model"),
    (4, 0, 6, 1, "mlx_vae"),
    (5, 0, 6, 0, "latents"),
    (6, 0, 7, 0, "images"),
    (7, 0, 8, 0, "IMAGE"),
    (7, 2, 8, 2, "AUDIO"),
    (8, 0, 9, 0, "VIDEO"),
    (2, 0, 5, 1, "condition"),
    (2, 0, 5, 2, "condition"),
]


def write_workflow(name, wf_id, title, nodes, links):
    workflow = build_workflow(wf_id, title, nodes, links)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[gen] {path.relative_to(ROOT)}  节点 {len(workflow['nodes'])} 条 / 连线 {len(workflow['links'])} 条")


# --- 1. 单图 → 视频（首帧锚点，Base 权重）------------------------------------------
def vae_loader(checkpoint: str = CKPT_BASE):
    """视频 VAE（把锚点图编码成 latent 行）；两档权重共用同一份 `MiniMax-H3`。"""
    return node_spec(
        10, "MlxVAELoader", "H3 视频 VAE（锚点编码，role=vae）",
        [], [["vae", "mlx_vae"]],
        ["minimax_h3", CKPT_BASE, "bfloat16", 8, "vae"], [0, 800], [380, 220], 3,
    )


PROMPT_SINGLE = (
    "integrated_multimodal_description: 视频共 1 个镜头，参考给出的那张海边栈桥照片："
    "镜头从栈桥尽头缓缓推向海平面，海鸥掠过画面前景，夕阳把木板的纹理照成暖橙色，"
    "最后定格在远处驶过的渔船。自然光、低饱和暖调，缓慢推镜带轻微手持呼吸感。\n\n"
    "overall_soundscape: 海浪反复拍岸、海鸥零星叫声、木栈桥被踩到的吱呀声。\n\n"
    "non_diegetic_music: 极简木吉他与弦乐铺底，节奏舒缓，音量克制。"
)


def workflow_single_image():
    """单图 → 视频：1 张图钉成首帧锚点（I2VA，用 Base 权重就够了）。"""
    nodes = base_nodes(PROMPT_SINGLE, visual_id=12, visual_label="首帧") + [
        vae_loader(),
        node_spec(11, "LoadImage", "H3 首帧（将拉伸到 640×352）",
                  [], [["IMAGE", "IMAGE"], ["MASK", "MASK"]],
                  ["h3_first_frame.png"], [420, 800], [360, 400], 4),
        node_spec(12, "MlxH3KeyframeCondition", "H3 首帧条件（LANCZOS 适配目标画布）",
                  [["vae", "mlx_vae"], ["first_frame", "IMAGE"], ["last_frame", "IMAGE"]],
                  [["keyframes", "mlx_h3_keyframes"], ["report", "STRING"]],
                  [640, 352], [820, 800], [380, 300], 5),
    ]
    links = list(BASE_LINKS) + [
        (10, 0, 12, 0, "mlx_vae"),
        (11, 0, 12, 1, "IMAGE"),
        (12, 0, 2, 1, "mlx_h3_keyframes"),
    ]
    write_workflow("minimax-h3-single-image-to-video", "mlx-minimax-h3-single-image-to-video",
                   "MiniMax-H3 单图生视频（首帧锚点 · Base 权重）", nodes, links)


# --- 2. 多图参考 → 视频（Ref2VA）/ 3. 纯参考（不钉锚点）----------------------------
def workflow_multi_image(only_reference=False):
    """参考生视频（Ref2VA）：3 张参考图，transformer 必须是 `MiniMax-H3-ref`。

    only_reference=False 时把第 1 张钉成首帧锚点（REF 也支持把 <Picture N>
    当具体帧锚点，其余两张只进 presentation）；True 时三张都只进 presentation、
    完全不占 latent 行。两种都是「按参考生成」，用 Base 权重都跟不出参考内容。
    """
    label = "3 张参考图（不钉锚点）" if only_reference else "3 张参考图（第 1 张钉首帧）"
    prompt = PROMPT_REF_ONLY if only_reference else PROMPT_MULTI
    nodes = base_nodes(prompt, visual_id=14, visual_label=label, checkpoint=CKPT_REF)
    images = [("h3_ref_1.png", 800), ("h3_ref_2.png", 1240), ("h3_ref_3.png", 1680)]
    if not only_reference:
        nodes.append(vae_loader(CKPT_REF))
    for index, (filename, y) in enumerate(images, start=1):
        nodes.append(
            node_spec(16 + index, "LoadImage", f"参考图 {index}",
                      [], [["IMAGE", "IMAGE"], ["MASK", "MASK"]],
                      [filename], [420, y], [360, 400], 4)
        )
    nodes += [
        node_spec(13, "MlxRefImageSet", f"{len(images)} 张参考图 → 有序图集",
                  [[f"image{index}", "IMAGE"] for index in range(1, 5)],
                  [["ref_source", "mlx_ref_image_src"], ["count", "INT"], ["report", "STRING"]],
                  [], [820, 1040], [380, 300], 5),
        node_spec(14, "MlxH3MultiReferenceCondition",
                  "H3 多图参考（3 张只进 presentation，不占 latent 行；要用 H3-REF）"
                  if only_reference
                  else "H3 多图参考（第 1 张钉首帧，其余只作参考；要用 H3-REF）",
                  [["ref_images", "mlx_ref_image_src"], ["vae", "mlx_vae"]],
                  [["keyframes", "mlx_h3_keyframes"], ["report", "STRING"]],
                  ["none", False, 640, 352] if only_reference else ["first", False, 640, 352],
                  [1240, 1040], [420, 320], 6),
    ]
    links = list(BASE_LINKS) + [
        *[(16 + index, 0, 13, index - 1, "IMAGE") for index in range(1, len(images) + 1)],
        # 图集必须真的接到条件的 ref_images（这是必填入口，漏接等于图片白load）
        (13, 0, 14, 0, "mlx_ref_image_src"),
        (14, 0, 2, 1, "mlx_h3_keyframes"),
    ]
    if not only_reference:
        links.append((10, 0, 14, 1, "mlx_vae"))
    name = "minimax-h3-reference-only-to-video" if only_reference else "minimax-h3-multi-image-to-video"
    title = (
        "MiniMax-H3 纯参考图生视频（H3-REF · 3 张只作参考，不钉锚点）" if only_reference
        else "MiniMax-H3 多图参考生视频（H3-REF · 3 张参考图，第 1 张钉成首帧）"
    )
    write_workflow(name, f"mlx-{name}", title, nodes, links)


# --- 4/5/6. 视频 → 视频（源视频帧当锚点；原声保留 / 丢弃 / 替换）------------------
def workflow_video_continuation(mode):
    """视频 → 视频：源视频的首 / 末帧当锚点。

    keep     = 原片段 + 新片段拼成成片，整段音轨 = 原声 + 生成音轨；
    drop     = 只出新生成的片段，原声直接丢弃（用 H3 生成的音轨）；
    replace  = 只出新生成的片段，音轨换成外部配乐。
    """
    prompts = {"keep": PROMPT_VIDEO_KEEP, "drop": PROMPT_VIDEO_DROP, "replace": PROMPT_VIDEO_REPLACE}
    labels = {
        "keep": "源视频续写（保留原声）",
        "drop": "源视频续写（丢弃原声）",
        "replace": "源视频续写（替换音轨）",
    }
    suffixes = {"keep": "keep-audio", "drop": "drop-audio", "replace": "replace-audio"}
    nodes = base_nodes(prompts[mode], visual_id=12, visual_label=labels[mode]) + [
        vae_loader(),
        node_spec(11, "LoadVideo", "源视频（只取一帧当锚点）",
                  [], [["VIDEO", "VIDEO"]],
                  ["h3_source_clip.mp4"], [420, 800], [380, 260], 4),
        node_spec(12, "MlxH3VideoCondition", "H3 视频条件（源视频 → 锚点）",
                  [["video", "VIDEO"], ["vae", "mlx_vae"]],
                  [["keyframes", "mlx_h3_keyframes"], ["report", "STRING"]],
                  ["continue_from_end", False, 640, 352], [820, 800], [420, 320], 5),
    ]
    chain_links = [
        (11, 0, 12, 0, "VIDEO"),
        (10, 0, 12, 1, "mlx_vae"),
        (12, 0, 2, 1, "mlx_h3_keyframes"),
    ]
    if mode == "keep":
        nodes += [
            node_spec(13, "GetVideoComponents", "拆出源视频的帧与音轨",
                      [["video", "VIDEO"]],
                      [["images", "IMAGE"], ["audio", "AUDIO"], ["fps", "FLOAT"],
                       ["bit_depth", "COMBO"], ["color_space", "COMBO"]],
                      [], [420, 1120], [380, 300], 5),
            node_spec(14, "CreateVideo", "原片段（原声；接进拼接节点第 1 路）",
                      [["images", "IMAGE"], ["fps", "FLOAT"], ["audio", "AUDIO"],
                       ["bit_depth", "COMBO"], ["color_space", "COMBO"],
                       ["codec", "COMBO"]],
                      [["VIDEO", "VIDEO"]],
                      [24, 8, "sRGB", "none"], [820, 1200], [380, 300], 6),
            node_spec(15, "AudioConcat", "原声 + 生成音轨 → 整段音轨",
                      [["audio1", "AUDIO"], ["audio2", "AUDIO"]],
                      [["AUDIO", "AUDIO"]],
                      ["after"], [820, 1560], [380, 260], 6),
            node_spec(16, "ConcatenateVideo", "原片段 + 新片段 → 成片",
                      [["video1", "VIDEO"], ["video2", "VIDEO"], ["complete_audio", "AUDIO"]],
                      [["VIDEO", "VIDEO"]],
                      ["auto"], [1240, 1200], [400, 260], 7),
        ]
        chain_links += [
            (11, 0, 13, 0, "VIDEO"),
            (13, 0, 14, 0, "IMAGE"),
            (13, 1, 14, 2, "AUDIO"),
            (13, 2, 14, 1, "FLOAT"),
            (13, 1, 15, 0, "AUDIO"),
            (7, 2, 15, 1, "AUDIO"),
            (14, 0, 16, 0, "VIDEO"),
            (8, 0, 16, 1, "VIDEO"),
            (15, 0, 16, 2, "AUDIO"),
            (16, 0, 9, 0, "VIDEO"),
        ]
        links = chain_links + [link for link in BASE_LINKS if link[2] != 9]
    elif mode == "replace":
        nodes.append(
            node_spec(13, "LoadAudio", "外部配乐（替换整段音轨）",
                      [], [["AUDIO", "AUDIO"]],
                      ["h3_bgm.mp3"], [420, 1160], [380, 240], 5)
        )
        links = (
            chain_links
            + [link for link in BASE_LINKS if link != (7, 2, 8, 2, "AUDIO")]
            + [(13, 0, 8, 2, "AUDIO")]
        )
    else:
        links = chain_links + list(BASE_LINKS)
    suffix = suffixes[mode]
    write_workflow(f"minimax-h3-video-continuation-{suffix}",
                   f"mlx-minimax-h3-video-continuation-{suffix}",
                   f"MiniMax-H3 视频续写（{labels[mode]}）",
                   nodes,
                   links)


# --- 7. 完整参考视频 → 动作迁移（Ref2VA）-------------------------------------------
def workflow_motion_reference():
    """完整参考视频：presentation + Video VAE latent，必须使用 H3-REF。"""
    nodes = base_nodes(
        PROMPT_MOTION_TRANSFER,
        sampler_wh=(640, 352),
        checkpoint=CKPT_REF,
        visual_id=12,
        visual_label="完整参考视频（动作 / 运镜）",
        steps=50,
        scheduler="minimax_h3",
    ) + [
        vae_loader(),
        node_spec(
            11,
            "LoadVideo",
            "动作参考视频（建议 5 秒以上，人物动作清晰）",
            [],
            [["VIDEO", "VIDEO"]],
            ["h3_motion_reference.mp4"],
            [420, 800],
            [380, 260],
            4,
        ),
        node_spec(
            12,
            "MlxH3MotionReferenceCondition",
            "H3 完整动作参考（2 fps presentation + 完整 Video VAE latent；必须 H3-REF）",
            [["video", "VIDEO"], ["vae", "mlx_vae"]],
            [["keyframes", "mlx_h3_keyframes"], ["report", "STRING"]],
            [640, 352, 124, True, 2.0],
            [820, 800],
            [460, 360],
            5,
        ),
    ]
    links = list(BASE_LINKS) + [
        (10, 0, 12, 1, "mlx_vae"),
        (11, 0, 12, 0, "VIDEO"),
        (12, 0, 2, 1, "mlx_h3_keyframes"),
    ]
    write_workflow(
        "minimax-h3-motion-transfer",
        "mlx-minimax-h3-motion-transfer",
        "MiniMax-H3 动作迁移（完整参考视频 · Ref2VA）",
        nodes,
        links,
    )


def workflow_motion_reference_with_image():
    """一张外观参考图 + 完整动作参考视频：图片与视频共同进入 Ref2VA。"""
    nodes = base_nodes(
        PROMPT_MOTION_TRANSFER_WITH_IMAGE,
        sampler_wh=(640, 352),
        checkpoint=CKPT_REF,
        visual_id=12,
        visual_label="1 张外观参考图 + 完整动作视频",
        steps=50,
        scheduler="minimax_h3",
    ) + [
        vae_loader(),
        node_spec(
            11,
            "LoadImage",
            "人物 / 外观参考图（提示词里的 <Picture 1>）",
            [],
            [["IMAGE", "IMAGE"], ["MASK", "MASK"]],
            ["h3_subject_reference.png"],
            [420, 760],
            [360, 400],
            4,
        ),
        node_spec(
            13,
            "LoadVideo",
            "动作与运镜参考视频（提示词里的 <Video 1>）",
            [],
            [["VIDEO", "VIDEO"]],
            ["h3_motion_reference.mp4"],
            [420, 1220],
            [380, 260],
            4,
        ),
        node_spec(
            12,
            "MlxH3MotionReferenceWithImageCondition",
            "H3 图片 + 动作迁移（<Picture 1> 外观 + <Video 1> 动作 / 运镜；必须 H3-REF）",
            [
                ["image", "IMAGE"],
                ["video", "VIDEO"],
                ["width", "INT"],
                ["height", "INT"],
                ["num_frames", "INT"],
                ["use_source_aspect", "BOOLEAN"],
                ["presentation_fps", "FLOAT"],
                ["vae", "mlx_vae"],
            ],
            [["keyframes", "mlx_h3_keyframes"], ["report", "STRING"]],
            [640, 352, 124, True, 2.0],
            [820, 960],
            [500, 420],
            5,
        ),
    ]
    links = list(BASE_LINKS) + [
        (10, 0, 12, 7, "mlx_vae"),
        (11, 0, 12, 0, "IMAGE"),
        (13, 0, 12, 1, "VIDEO"),
        (12, 0, 2, 1, "mlx_h3_keyframes"),
    ]
    write_workflow(
        "minimax-h3-motion-transfer-with-image",
        "mlx-minimax-h3-motion-transfer-with-image",
        "MiniMax-H3 图片动作迁移（1 张外观参考图 + 完整动作视频 · Ref2VA）",
        nodes,
        links,
    )


# --- 7. 全能参考生视频（4 张参考图 + 加速 LoRA，Ref2VA）---------------------------
def workflow_all_reference():
    """对照社区那份「3 步加速 Lora + 全能参考」的工作流，做一份只靠本插件节点的版本。

    与那份参考工作流的差别（都是节点能力决定的，不是偷懒）：

    * 参考图：本插件最多 4 个槽位（`MlxRefImageSet` 只有 image1..image4），
      参考工作流里的 9 张 / 3 段参考视频 / 3 段参考音频都塞不进同一条链 ——
      H3 只有一条 keyframes 链（`MlxH3KeyframeCondition` /
      `MlxH3MultiReferenceCondition` / `MlxH3VideoCondition` 三选一），
      且只有「多图参考」这一路能吃多张图；想换源视频续写请用
      `minimax-h3-video-continuation-*.json`，想把音轨换成外部配乐
      请用 `minimax-h3-video-continuation-replace-audio.json`。
    * 采样：不需要参考工作流里的 `KSamplerSelect` / `BasicScheduler` /
      `BasicGuider` / `SamplerCustomAdvanced` 四件套 —— 「MLX 采样器」自己
      写采样循环，加速适配器只要在「MLX 模型 LoRA」上选对文件、
      再把 steps 改成适配器标称的步数即可。
    * 加速 LoRA 选 `..._ref2v_lightx2v_turbo_4step_...`（文件名标明 Ref2VA，4 步）；
      不照抄参考工作流里的 `..._taomate_3step_...`（文件名只写步数、没写任务）
      与 `..._fl2v_...`（那是首 / 尾帧任务的），免得任务跟 REF 权重对不上。
    """
    nodes = base_nodes(
        PROMPT_ALL_REF,
        sampler_wh=(640, 352),
        checkpoint=CKPT_REF,
        visual_id=16,
        visual_label="4 张参考图（不钉锚点）",
        steps=4,
        scheduler="minimax_h3",
    ) + [
        node_spec(17, "MlxModelLoraApply", "Ref2VA 加速 LoRA（4 步，强度 1.0）",
                  [["model", "model"], ["lora", "COMBO"], ["strength", "FLOAT"]],
                  [["model", "model"]],
                  [LORA_H3_REF_4STEP, 1.0], [420, 560], [380, 240], 2),
    ]
    for index, filename in enumerate(
        ("h3_ref_1.png", "h3_ref_2.png", "h3_ref_3.png", "h3_ref_4.png"), start=1
    ):
        nodes.append(
            node_spec(10 + index, "LoadImage", f"参考图 {index}（提示词里的 <Picture {index}>）",
                      [], [["IMAGE", "IMAGE"], ["MASK", "MASK"]],
                      [filename], [420, 880 + 440 * index], [360, 400], 4)
        )
    nodes += [
        node_spec(15, "MlxRefImageSet", "4 张参考图 → 有序图集（占满 4 个槽位）",
                  [[f"image{index}", "IMAGE"] for index in range(1, 5)],
                  [["ref_source", "mlx_ref_image_src"], ["count", "INT"], ["report", "STRING"]],
                  [], [820, 1760], [380, 300], 5),
        node_spec(16, "MlxH3MultiReferenceCondition",
                  "H3 多图参考（4 张只进 presentation，不占 latent 行；必须用 H3-REF）",
                  [["ref_images", "mlx_ref_image_src"], ["vae", "mlx_vae"]],
                  [["keyframes", "mlx_h3_keyframes"], ["report", "STRING"]],
                  ["none", False, 640, 352], [1240, 1760], [420, 320], 6),
    ]
    links = [
        # 模型链改走加速 LoRA：transformer → LoRA → 采样器（其余连线沿用骨架）
        *[link for link in BASE_LINKS if link != (3, 0, 5, 0, "model")],
        (3, 0, 17, 0, "model"),
        (17, 0, 5, 0, "model"),
        *[(11 + index, 0, 15, index, "IMAGE") for index in range(4)],
        # 图集必须真的接到条件的 ref_images（必填入口，漏接等于图片白 load）
        (15, 0, 16, 0, "mlx_ref_image_src"),
        (16, 0, 2, 1, "mlx_h3_keyframes"),
    ]
    write_workflow("minimax-h3-all-reference-to-video", "mlx-minimax-h3-all-reference-to-video",
                   "MiniMax-H3 全能参考生视频（H3-REF · 4 张参考图 + 4 步加速 LoRA）",
                   nodes, links)


if __name__ == "__main__":
    workflow_single_image()
    workflow_multi_image()
    workflow_multi_image(only_reference=True)
    workflow_video_continuation("keep")
    workflow_video_continuation("drop")
    workflow_video_continuation("replace")
    workflow_motion_reference()
    workflow_motion_reference_with_image()
    workflow_all_reference()
    print("[gen] 九份 H3 工作流已生成")
