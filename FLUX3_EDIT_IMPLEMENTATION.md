# FLUX 3 Image Klein 单图 / 多图编辑实现说明

> 状态：**代码与契约测试已落地**。本说明记录 FLUX 3 与 Flux.2 不同的模型路由、
> token 顺序、参考图预处理和工作流连线规则。
>
> 当前已通过无真实大模型权重的 guard 测试、编译检查和 diff 检查；真实官方权重的
> 单图与多图 Metal 端到端视觉结果尚未作为本仓库的最终验收记录。因此本文不会把
> “实现契约测试通过”表述成“已完成真实出图验收”。

## 1. 适用范围

FLUX 3 Image Klein 在本项目中是独立的 `flux3` family，不复用 Flux.2 的 Transformer
或编辑采样分支。当前工作流包括：

```text
workflows/flux3-klein-base.json
workflows/flux3-klein-distilled.json
workflows/flux3-klein-edit.json
workflows/flux3-klein-edit-multi.json
```

四个工作流分别覆盖 Base 文生图、distilled 文生图、单图参考编辑和有序多图参考编辑。
导入后需要把三个 Loader 中的模型目录改成本机实际扫描到的目录名。

### 1.1 模型目录

FLUX 3 的官方/Diffusers 风格 checkpoint 需要按组件放到 MLX 模型根目录：

```text
/Users/apple/ComfyUI-Shared/models/mlx/
├── transformer/flux-3-image-klein/
├── text_encoder/flux-3-image-klein/
└── vae/flux-3-image-klein/
```

Base checkpoint 可以使用对应的 `flux-3-image-klein-base` 目录。三个 Loader 的
`model_type` 都必须是 `flux3`，Transformer、text encoder 和 VAE 必须来自同一份
checkpoint。FLUX 3 的 tokenizer 由 text encoder 组件解析；不要将 FLUX 3 误接到
Flux.2 的 tokenizer/Transformer 契约上。

`types.py` 会从模型名或本地 `config.json` 检测 FLUX 3。检测到 FLUX 3 时，选择
`flux2` 会直接报错，而不是静默加载错误的 Flux.2 结构。

## 2. 工作流连线

### 2.1 文生图

```text
MlxClipLoader(flux3)
  ├─→ MlxTextEncoder（正向）─┐
  └─→ MlxTextEncoder（负向）─┴→ MlxKSamplerMLX
MlxTransformerLoader(flux3) ───────────────┘
MlxVAELoader(flux3) ─→ MlxVAEDecoder
```

distilled 工作流使用 `flux3_flow_map` 和无 CFG 的推荐设置；Base 工作流保留 CFG
条件。实际步数、guidance 和 scheduler 以对应 JSON 工作流的 widget 为准，不要把
distilled checkpoint 与 Base 工作流的参数混用。

### 2.2 单图编辑

单图编辑的参考图只经过 VAE 编码，不进入文本编码器：

```text
LoadImage
    ── IMAGE ─→ MlxVAEEncoder.images
MlxVAELoader
    ├─ vae ─→ MlxVAEEncoder
    └─ vae ─→ MlxVAEDecoder
MlxVAEEncoder.ref_images ─→ MlxKSamplerMLX.ref_images
```

`MlxTextEncoder` 只编码文字指令。不要把 `MlxVAEEncoder.ref_images` 接到
`MlxTextEncoder.ref_images`；FLUX 3 会明确拒绝这条路径，因为参考图条件属于 DiT 的
`image_cond` 流，而不是文本编码器输入。

对应工作流：

```text
workflows/flux3-klein-edit.json
```

### 2.3 多图编辑

不同尺寸的多张参考图必须通过 `MlxRefImageSet`，而不是先用 ComfyUI 的 `Batch Images`
强行合并：

```text
Load Image 1 ──→ image1 ┐
Load Image 2 ──→ image2 ├─→ MlxRefImageSet.ref_source
Load Image 3 ──→ image3 │
Load Image 4 ──→ image4 ┘

MlxRefImageSet.ref_source ─→ MlxVAEEncoder.ref_source
MlxVAEEncoder.ref_images  ─→ MlxKSamplerMLX.ref_images
```

当前示例工作流使用前三张图，第四个槽位可选。`MlxRefImageSet` 的输入顺序是模型
语义的一部分：提示词中的“第一张 / 第二张 / 第三张”必须对应 `image1 / image2 /
image3`，不要在中间重新排序。`MlxVAEEncoder` 的 `images` 和 `ref_source` 两个
入口不能同时连接。

对应工作流：

```text
workflows/flux3-klein-edit-multi.json
```

## 3. 参考图预处理契约

FLUX 3 不使用 Flux.2 的“每张参考图缩放到约 1MP”规则。当前
`prepare_flux3_reference_image()` 的处理顺序是：

1. 转换为 RGB；
2. 只有当单张图片面积超过 `4096 × 4096` 时，才按比例缩小到不超过该面积；
3. 将宽和高分别中心裁剪到 16 的倍数；
4. 保留每张参考图自己的宽高和宽高比。

因此，正常尺寸的 `1600×800` 参考图仍是 `1600×800`，不会被缩放到约 1MP；
`1611×803` 会中心裁剪为 `1600×800`。参考图至少需要每边能裁出一个 16 像素块。

`resize_mode` 对 FLUX 3 的含义：

| 值 | 行为 |
| --- | --- |
| `auto` | 按 Flux3 的面积上限与 16 像素网格处理 |
| `aspect_area_crop` | 同上，适合示例工作流的显式设置 |
| `keep_resolution` | 不做 `4096²` 面积缩放，但仍转 RGB 并中心裁到 16 的倍数；可能明显增加内存 |

## 4. latent、ID 与 token 顺序

### 4.1 latent 形状

Flux VAE 的 32 通道 latent 经过 2×2 patchify 后变成 128 通道 token：

```text
VAE latent:       [B, 32, H/8, W/8]
patchify:         [B, 128, H/16, W/16]
pack:             [B, (H/16)*(W/16), 128]
```

目标图和参考图都使用这一 128 通道 image token 契约。目标图的 packed latent 进入
`image` 流，参考图 packed latent 按输入顺序拼接进入 `image_cond` 流。

### 4.2 grid IDs

目标图的 layer 坐标为 `0`。多图参考按输入顺序使用负 layer 坐标：

```text
参考图 1 → l = -1
参考图 2 → l = -2
参考图 3 → l = -3
...
```

参考图的输入顺序同时决定：

- VAE 编码后 token 的 concat 顺序；
- `image_cond_ids` 的 layer 坐标；
- 提示词中“第一张、第二张……”的语义对应关系。

### 4.3 joint sequence 顺序

FLUX 3 的 mode blocks 之后，single/joint blocks 的正式顺序是：

```text
[text, image, image_cond]
```

这不是 Flux.2 的 `[image, image_cond, text]` 顺序。实现保留三个 mode 的独立索引，
但在进入 joint blocks 前严格按上面的顺序拼接，并在同一顺序下拼接 joint RoPE IDs
和 single-stream modulation。只取回 `image` 段作为目标图预测结果；`image_cond` 段
不会被当作输出 latent。

## 5. 缓存与条件路径

- 参考图编码结果存入 `ref_encoding` 缓存；节点 handle 只保存缓存键和元信息。
- 参考图输入顺序、尺寸和像素变化会改变参考图缓存键。
- 单图/多图参考条件由 `MlxKSamplerMLX.ref_images` 消费。
- FLUX 3 文本正/负条件仍由两个普通 `MlxTextEncoder` 节点生成。
- sampler 不修改 Flux.2 的采样逻辑；FLUX 3 在 `pipeline.py` 中走独立的 FLUX 3
  采样分支。

## 6. 验证状态

当前仓库可用以下命令执行依赖轻量的契约测试：

```bash
cd /Users/apple/workspace/python/ComfyUI-MLX-GEN
PYTHONPATH=src /Users/apple/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/bin/python \
  -m pytest -q tests/test_flux3_guard.py tests/test_lora.py
```

已覆盖的 FLUX 3 约束包括：

- `flux3` 是独立 family，不能回退到 Flux.2；
- packed latent 与 transformer 默认层数/通道契约；
- `[text, image, image_cond]` joint 顺序；
- 正常尺寸参考图不执行 Flux.2 的约 1MP 缩放；
- 参考图输入顺序对应 `l=-1,-2,...`；
- Flux3 编辑工作流不把参考图接入文本编码器。

编译和空白检查：

```bash
PYTHONPATH=src /Users/apple/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/bin/python \
  -m compileall -q src/comfyui_mlx_gen/flux3 \
  src/comfyui_mlx_gen/pipeline.py tests/test_flux3_guard.py
git diff --check
```

真实权重出图时，建议先用 `flux3-klein-edit.json` 验证单图，再用三张视觉内容明显
不同的图片验证 `flux3-klein-edit-multi.json`，并检查日志中的参考图数量、packed token
数量和顺序。真实权重的视觉 A/B 结果仍应单独记录，不能用本 guard 测试代替。

## 7. 与 Flux.2 的区别速查

| 项目 | Flux.2 | FLUX 3 |
| --- | --- | --- |
| model family | `flux2` | `flux3` |
| 参考图尺寸策略 | 约 1MP 等比处理 | 仅超过 `4096²` 才缩小，随后对齐 16 |
| 参考图 layer 坐标 | Flux.2 的时间/网格约定 | 输入顺序对应 `-1,-2,...` |
| joint 顺序 | `[image, image_cond, text]` | `[text, image, image_cond]` |
| 文本编码器接参考图 | Flux.2 工作流另有 family 约定 | **不接**，只进 `MlxKSamplerMLX.ref_images` |
| 多图异尺寸入口 | `MlxRefImageSet` + Flux.2 规则 | `MlxRefImageSet` + FLUX 3 规则 |

最重要的操作原则是：不要把 Flux.2 的工作流、模型目录或参考图预处理参数直接套到
FLUX 3 上；请从 `workflows/flux3-klein-*.json` 开始，并保持三个 Loader 的
`model_type=flux3`。