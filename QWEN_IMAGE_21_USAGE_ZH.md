# Qwen-Image 2.1 下载、安装与使用指南

本文说明如何在 **Apple Silicon + ComfyUI-MLX-GEN** 中使用官方
[`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) 权重进行：

- 文生图；
- 单张参考图编辑；
- 最多 10 张、有顺序且尺寸可以不同的参考图编辑；
- RGBA 透明图片生成和保存；
- 按文件夹逐张进行图生图，并将图片中的文字替换为中文。

这里的 `qwen_image_21` 是 Qwen-Image 2.1 的原生统一生成/编辑链路，不是旧版
`qwen_image`（Qwen-Image 2512 文生图）或 `qwen_edit`（Qwen-Image-Edit 2511）。三个
Loader 的 `model_type` 必须全部选择 `qwen_image_21`。

## 1. 前置条件

- Apple Silicon Mac（M 系列芯片）；
- 已安装并可启动 ComfyUI；
- 本插件位于：

  ```text
  <ComfyUI>/custom_nodes/ComfyUI-MLX-GEN
  ```

- 已使用 **ComfyUI 实际运行的 Python** 安装插件依赖：

  ```bash
  cd /你的/ComfyUI/custom_nodes/ComfyUI-MLX-GEN
  /你的/ComfyUI/.venv/bin/python -m pip install -r requirements.txt
  ```

官方仓库采用 Qwen Research License。下载和使用前请阅读模型页上的许可证条款。官方
BF16 仓库下载后约占 **31 GB**：Transformer 约 13 GB、Qwen3-VL 文本/视觉编码器约
16 GB、VAE 约 1.3 GB，另外还有 processor 配置文件。请预留额外空间给下载缓存和输出。

## 2. 模型根目录

插件通过 ComfyUI 的 `folder_paths` 获取注册为 `mlx` 的模型根目录。当前 ComfyUI Desktop
配置使用共享目录：

```text
/Users/apple/ComfyUI-Shared/models/mlx
```

ComfyUI Desktop 启动时会通过 `extra_model_paths` 注册：

```text
base_path: /Users/apple/ComfyUI-Shared/models
mlx: mlx/
```

代码使用 `folder_paths.get_folder_paths("mlx")` 获取该目录，不会根据插件安装位置寻找
模型。模型数据也可以放在外置磁盘；只需让下文的四个组件目录使用绝对软链接指向外置
磁盘，无需修改 `paths.py`。

## 3. 下载官方模型

### 3.1 使用 Hugging Face CLI（推荐）

先安装 Hugging Face CLI。可以使用普通终端 Python，也可以使用 ComfyUI 的 Python：

```bash
python3 -m pip install -U huggingface_hub
```

如果下载时提示需要身份验证，先登录；公开访问正常时可以跳过这一步：

```bash
hf auth login
```

把完整仓库下载到一个有至少 35 GB 可用空间的位置。下面以用户目录为例：

```bash
mkdir -p "$HOME/Models/Qwen-Image-2.1"
hf download Qwen/Qwen-Image-2.1 \
  --local-dir "$HOME/Models/Qwen-Image-2.1"
```

下载中断后可以直接重新执行相同命令；CLI 会复用已经完成的文件。

### 3.2 不使用 CLI：Python 下载

如果终端找不到 `hf` 命令，可以使用同一个包的 Python API：

```bash
python3 - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

destination = Path.home() / "Models" / "Qwen-Image-2.1"
snapshot_download(
    repo_id="Qwen/Qwen-Image-2.1",
    local_dir=destination,
)
print(f"模型已下载到：{destination}")
PY
```

### 3.3 检查下载是否完整

至少应存在以下文件。权重 shard 名称和数量以官方仓库当前版本为准，不要只下载
`config.json` 或 Git/Xet 指针文件：

```text
Qwen-Image-2.1/
├── transformer/
│   ├── config.json
│   ├── diffusion_pytorch_model-00001-of-00002.safetensors
│   ├── diffusion_pytorch_model-00002-of-00002.safetensors
│   └── diffusion_pytorch_model.safetensors.index.json
├── text_encoder/
│   ├── config.json
│   ├── model-00001-of-00004.safetensors
│   ├── ...
│   └── model.safetensors.index.json
├── processor/
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── ...
└── vae/
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

可用下面的命令快速检查：

```bash
MODEL="$HOME/Models/Qwen-Image-2.1"
test -f "$MODEL/transformer/config.json" && \
test -f "$MODEL/text_encoder/config.json" && \
test -f "$MODEL/processor/tokenizer.json" && \
test -f "$MODEL/vae/config.json" && \
echo "Qwen-Image 2.1 目录检查通过"
du -sh "$MODEL"
```

## 4. 把模型放到插件能扫描的位置

插件按组件扫描四个目录，四处显示的模型名必须同为 `Qwen-Image-2.1`：

```text
/Users/apple/ComfyUI-Shared/models/mlx/
├── transformer/Qwen-Image-2.1/ -> <下载目录>/transformer
├── text_encoder/Qwen-Image-2.1/ -> <下载目录>/text_encoder
├── tokenizer/Qwen-Image-2.1/ -> <下载目录>/processor
└── vae/Qwen-Image-2.1/ -> <下载目录>/vae
```

> **注意：**官方仓库把 tokenizer 文件放在 `processor/`，而插件的组件目录名是
> `tokenizer/`。因此第三个链接必须是“本地 `tokenizer` → 官方 `processor`”，不能链接到
> 一个不存在的官方 `tokenizer/` 目录。

推荐使用软链接，约 31 GB 的权重只保留一份：

```bash
MLX_ROOT="/Users/apple/ComfyUI-Shared/models/mlx"
SNAPSHOT="$HOME/Models/Qwen-Image-2.1"

mkdir -p "$MLX_ROOT"/{transformer,text_encoder,tokenizer,vae}

# 先把源目录规范成绝对路径，避免移动工作目录后软链接失效。
SNAPSHOT="$(cd "$SNAPSHOT" && pwd)"

ln -sfn "$SNAPSHOT/transformer"  "$MLX_ROOT/transformer/Qwen-Image-2.1"
ln -sfn "$SNAPSHOT/text_encoder" "$MLX_ROOT/text_encoder/Qwen-Image-2.1"
ln -sfn "$SNAPSHOT/processor"    "$MLX_ROOT/tokenizer/Qwen-Image-2.1"
ln -sfn "$SNAPSHOT/vae"          "$MLX_ROOT/vae/Qwen-Image-2.1"
```

检查四个链接：

```bash
for component in transformer text_encoder tokenizer vae; do
  test -e "$MLX_ROOT/$component/Qwen-Image-2.1" \
    && echo "OK: $component" \
    || echo "缺失或软链接断开: $component"
done
```

也可以把四个真实组件目录复制到对应位置，但不要把整个 snapshot 只复制成
`/Users/apple/ComfyUI-Shared/models/mlx/Qwen-Image-2.1`；Loader 不会扫描这个层级。
放好模型后要**完整重启
ComfyUI**，浏览器刷新不会重新生成 Loader 下拉列表。

## 5. 文生图

1. 启动 ComfyUI。
2. 导入：

   ```text
   workflows/qwen-image-2.1.json
   ```

3. 检查三个 Loader：

   | 节点 | 关键设置 |
   | --- | --- |
   | MLX CLIP 加载 | `model_type=qwen_image_21`、`component=text_encoder`、`path=Qwen-Image-2.1`、`quantize=8` |
   | MLX 模型加载 | `model_type=qwen_image_21`、`model_path=Qwen-Image-2.1`、`quantize=8`、`compile=false` |
   | MLX VAE 加载 | `model_type=qwen_image_21`、`model_path=Qwen-Image-2.1`、`quantize=0`、`role=vae` |

   官方下载的是 BF16 权重。这里的 `quantize=8` 表示插件在加载文本编码器和 Transformer
   时逐 shard 在线转换为 MLX q8，以降低统一内存占用；不会改写磁盘上的官方文件。VAE 是
   卷积网络，保持 `quantize=0`。

   如果要保留 Qwen-Image 2.1 文本编码器的 16bit 精度，在 `MLX CLIP 加载` 中选择
   `precision=MLX 16bit`，并把该节点的 `quantize` 设为 `0`。`MLX 16bit` 在内部映射为
   MLX `float16`；`quantize` 是独立开关，仍设为 4 或 8 时会继续执行在线量化。

4. 在正向 `MLX 文本编码器` 中填写提示词。负向节点可以保留单个空格；当
   `guidance=1.0` 时负向条件不会参与采样。
5. 先保持示例参数进行验证：
   - 1024×1024；
   - 40 步；
   - `guidance=1.0`；
   - `scheduler=flow_match_euler_discrete`；
   - batch 1；
   - `compile=false`。
6. 点击 **Queue Prompt/执行**。

### 自定义采样尺寸

`MlxKSamplerMLX` 的 `width` 和 `height` 现在是可直接输入的整数，不再限制为下拉
档位。你可以在采样节点中填写例如 `1000 × 1000`、`1152 × 896`、`1536 × 1024`，
然后直接执行。

Qwen-Image 2.1 允许填写 `32–4096` 范围内的任意整数宽高，不要求是 **32 的倍数**。
由于模型内部的 latent 需要 2×2 block，采样时会把每边临时向上对齐到最近的 32 像素，
VAE 解码后再裁回你填写的最终尺寸。前端的步进设为 `32` 只是方便加减，并不会限制
手动输入。分辨率越大，统一内存占用和生成时间越高，建议先从 `1024 × 1024` 或
`1152 × 896` 开始测试。

官方模型卡以 2048×2048、40 步为示例，并列出 4:3、3:4、3:2、2:3、16:9 和 9:16 等
比例；本插件工作流为了先控制 Apple 统一内存占用，默认从 1024×1024 开始。确认正常后再
逐步提高分辨率。

## 5.1 文件夹逐张图生图：图片文字替换为中文

需要把一个文件夹中的图片逐张独立处理时，导入：

```text
workflows/qwen-image-2.1-folder-batch-edit.json
```

该工作流使用新的 **`Qwen-Image 2.1 文件夹逐张编辑（图片文字改为中文）`** 节点，
不使用旧的分镜批处理节点，也不会把整个目录作为一个多图参考批次。节点内部严格按下面的
顺序循环：

```text
读取 1 张图片
  → 1 张参考图 VAE 编码 + Qwen3-VL 条件编码
  → batch_size=1、ref_count=1 独立采样
  → VAE 解码 1 张
  → 保存 1 张
  → 处理下一张
```

使用步骤：

1. 点击节点里的 **选择输入文件夹**，在浏览器中选择本地图片文件夹。这里是文件夹选择器，
   不是输入路径；插件会把所选文件夹中的图片上传到 ComfyUI 的输入目录，并自动记录本次选择。
   不要在隐藏的 `input_directory` 状态值中填写本机路径。
2. 节点会使用你选择的文件夹名称作为 ComfyUI 输入目录下的目录名，并在这个文件夹下自动创建
   `image` 文件夹，把所有生成的 PNG 保存到这里，不需要单独填写输出路径。节点按钮会显示
   `输出：image/`。
3. 三个 Loader 都选择 `qwen_image_21` 和同一套 `Qwen-Image-2.1` 权重。
4. 节点会对每张图片自动读取原始宽度和高度，并提供 `分辨率系数` 及 `缩放时机`。该系数可以是小数，
   例如设置为 `0.5` 时，原图 `1920 × 1080` 的采样画布为 `960 × 540`。默认的
   **采样时降分辨率** 会让参考图仍以原始 `1920 × 1080` 传入 VAE 和 Qwen3-VL 条件编码，
   只把采样器的目标画布设为 `960 × 540`；不会先缩小传入的图片，也不会在解码后额外缩放。
   **先降分辨率** 会先把参考图缩小到 `960 × 540`，再用缩小后的参考图和同尺寸采样画布生成；
   这是旧工作流中“先降分辨率”的原有语义。选择 **后降分辨率** 时，参考图和采样都使用原图尺寸，
   最后才把生成结果缩放到 `960 × 540` 保存。默认系数 `1.0` 时三种模式的最终尺寸相同。
   参考图在“采样时降分辨率”和“后降分辨率”中使用保留原始分辨率模式；“先降分辨率”则明确使用缩小后的参考图。
5. 正向提示词默认要求保留人物和构图，并识别图片中的招牌、字幕、标签、包装文字和
   对话框，将所有文字替换为准确、自然、清晰、可读的简体中文，同时保持位置、字号层级、
   字体风格、颜色、方向和排版布局。没有文字的图片不会被额外添加文字。
6. 点击 **Queue Prompt/执行**。

执行期间，节点进度条会显示当前处理数量 / 总图片数量，例如 `3/20`。后台日志也会显示
`处理进度：3/20（成功 3，失败 0）`；每张图片成功保存或处理失败后都会推进一次，因此
失败的图片也会计入“已处理”，并继续处理后续图片。

输入目录只处理常见图片格式：`.png`、`.jpg`、`.jpeg`、`.webp`、`.bmp`、`.tif`、`.tiff`。
文件按自然排序处理，例如 `image1.png`、`image2.png`、`image10.png`，而不是按字符串排序。
每张图片使用 `seed + 序号`，因此同一次执行中的图片具有独立且可重复的随机种子。
每张图片的宽高独立自适应。使用 **采样时降分辨率** 时，分辨率系数计算出的任一采样边小于 32 像素，
该图片会处理失败并继续下一张；参考图仍按原始尺寸传入，因此参考图本身必须满足 Qwen-Image 2.1 的尺寸限制。
使用 **先降分辨率** 时，缩小后的参考图和采样画布都必须满足 Qwen-Image 2.1 的尺寸限制。
使用 **后降分辨率** 时，生成阶段使用原图尺寸，最终保存的 PNG 可以小于 32 像素。Qwen-Image 2.1
内部会将采样尺寸临时向上对齐到 32 像素边界，VAE 解码后再裁回采样阶段要求的宽高。系数越大，
统一内存占用和生成时间越高；需要降低显存/统一内存占用时，可以使用默认的 **采样时降分辨率** 和 `0.5` 或 `0.75`。

输出文件统一保存到 ComfyUI 输入目录中**你选择的文件夹**的 `image/` 子目录，格式为 PNG，命名格式为：

```text
0000_原文件名stem.png
0001_原文件名stem.png
...
```

例如选择本地文件夹 `my-images` 后，ComfyUI 服务端的实际路径为：

```text
<ComfyUI>/input/mlx_qwen_image_21/my-images/image/
```

其中 `image/` 不存在时会自动创建。浏览器不会把本机原始文件夹的绝对路径交给服务端，
所以结果写入的是 ComfyUI 的上传副本目录，而不是浏览器选择器所在电脑上的原目录。

节点的 `saved_files` 输出会列出实际保存路径，`report` 输出会给出每张图片的成功或失败信息，
`success_count` 输出为成功保存的数量。某一张图片失败时节点会继续处理后续图片；如果全部失败，
节点最终报错并显示完整报告。由于模型需要重新绘制文字，复杂字体、极小文字、严重遮挡或低清晰度
文字可能无法完全准确识别，建议先用清晰、分辨率较高的输入图片测试。

## 5.2 Qwen-Image 2.1 PE 提示词增强（MLX-VLM / Transformers）

`MlxQwenImagePET2I` 和 `MlxQwenImagePEI2I` 是独立的提示词重写节点，不是 Qwen-Image
主模型的 `text_encoder`。它们输出 `rewritten_prompt`，可以连接到正向
`MlxTextEncoder.prompt`。

本插件支持以下加载类型：

| 加载类型 | 适用目录 | 依赖 | `system_prompt.txt` |
| --- | --- | --- | --- |
| `MLX-VLM (4bit)` / `MLX-VLM (8bit)` | `Qwen-Image-2.1-PE-*-MLX` 的 `4bit/` 或 `8bit/` | `mlx-vlm` | 不要求；若同目录下安装了对应官方 Transformers PE，会自动复用其 system prompt |
| `MLX 16bit` | MLX checkpoint 的 `16bit/` 子目录；若不存在则使用 checkpoint 根目录的 FP16/BF16 权重 | `mlx-vlm` | 不要求；若同目录下安装了对应官方 Transformers PE，会自动复用其 system prompt |
| `Transformers` | 官方 `Qwen/Qwen-Image-2.1-PE-T2I` 或 `PE-I2I` snapshot | `torch`、`transformers` | 必须存在 |

你的 MLX checkpoint 应保持完整目录，不要只复制 safetensors：

```text
text_encoder/Image-2.1-PE-T2I-MLX/
├── 4bit/
│   ├── config.json
│   ├── chat_template.jinja
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── model.safetensors.index.json
│   └── model-*.safetensors
├── 8bit/                         # 可选
└── 16bit/                        # 可选；也可以把非量化权重直接放在根目录
```

节点下拉只扫描 `text_encoder/` 的直接子目录，不会把 `4bit/`、`8bit/`、`16bit/` 等嵌套目录显示为
独立模型。T2I 节点只能选择 `PE-T2I`，I2I 节点只能选择 `PE-I2I`；代码会在执行时再次校验，
因此旧工作流中的 stale dropdown 值不会被错误地送入另一种模型。I2I 节点还必须连接
`IMAGE`；T2I 节点不需要图片。

安装依赖后**完整重启 ComfyUI**，再在节点中选择 `model_path` 和 `loader_type`：

```bash
/你的/ComfyUI/.venv/bin/python -m pip install -r requirements.txt
```

如果选择 `Transformers`，缺少 `system_prompt.txt` 会得到明确的文件错误；如果使用
你当前的 `prithivMLmods/...-MLX` checkpoint，请根据权重目录选择 `MLX-VLM (4bit)`、
`MLX-VLM (8bit)` 或 `MLX 16bit`，不需要手工复制或修改 MLX snapshot。`MLX 16bit`
由 `mlx-vlm` 按 checkpoint 配置自动加载非量化 FP16/BF16 safetensors，不会执行额外量化。

### 5.1 生成透明 RGBA 图片

官方推荐在提示词中明确声明透明图，例如：

```text
This is an RGBA image with transparency. A cute cartoon dragon sticker.
The image has alpha channel and the background is transparent.
```

要保留 alpha 通道，请使用工作流中的 `MLX 保存图片（MlxSaveImage）` 输出 PNG。普通
ComfyUI `IMAGE` 预览链路主要用于 RGB 预览；透明度还可从 `MlxPilToTorch.MASK` 输出取得。

## 6. 单图编辑

1. 导入：

   ```text
   workflows/qwen-image-2.1-edit.json
   ```

2. 在 `Load Image` 中选择参考图。
3. 在正向 `MLX 文本编码器` 中写编辑指令，例如：

   ```text
   保持主体身份和服装细节不变，把背景替换为日落时的海滩，电影感光线。
   ```

4. 确认 `MLX VAE 编码.ref_images` 的**同一个输出**同时连接到：

   ```text
   MLX VAE 编码.ref_images ─┬→ 正向 MLX 文本编码器.ref_images
                            ├→ 负向 MLX 文本编码器.ref_images
                            └→ MLX 采样器.ref_images
   ```

5. 保持 40 步、guidance 1.0、flow-match 和 1024×1024，先执行一次。

Qwen-Image 2.1 与 legacy `qwen_edit` 不同：参考图允许和目标画布尺寸不同。`MLX VAE
编码`的 `auto` 模式会保持参考图比例、按目标面积缩放并对齐到 32 像素倍数；目标画布使用
接近参考图的宽高比，通常能减少构图偏移。

正向和负向文本编码器都要看到参考图，因为 Qwen3-VL 视觉条件属于提示词 prefix；采样器
还需要同一份参考 VAE latent。插件会比较三个缓存键，连错、漏连或换成另一批图时会在加载
DiT 前报错。

## 7. 多参考图编辑（最多 10 张）

直接导入示例工作流：

```text
workflows/qwen-image-2.1-edit-multi.json
```

示例已接入 3 张图，分别用于主体/身份与构图、服装/道具、背景/风格。可删除不用的
`Load Image`，也可继续连接 `MlxRefImageSet.image4` 到 `image10`；同时按实际用途修改提示词中
“第一张、第二张……”的指代。

参考图尺寸不同时，不要使用 ComfyUI 的 `Batch Images`，因为它会把后续图片调整到第一张
的尺寸。使用 `MLX 参考图集（MlxRefImageSet）`：

```text
Load Image 1 ─┐
Load Image 2 ─┼→ MlxRefImageSet.ref_source → MlxVAEEncoder.ref_source
Load Image 3 ─┘
```

然后仍把 `MlxVAEEncoder.ref_images` 同时接到正向条件、负向条件和采样器。注意：

- `image1 → image2 → ... → image10` 就是提示词中的第一张、第二张……，顺序不会重排；
- 每个输入槽也可以是同尺寸图片批次，批次内部按原顺序展开；
- 总数最多 10 张；`MlxVAEEncoder.max_reference_images` 应不小于实际张数；
- 使用 `ref_source` 时不要再连接 `MlxVAEEncoder.images`，两个入口只能二选一；
- 每张图分别保持宽高比并编码，不要求参考图之间尺寸相同；
- 提示词应明确各图用途，例如“保持第一张的人物身份，使用第二张的服装和第三张的背景”。

## 8. 内存与速度建议

- 首次从官方 BF16 checkpoint 加载并在线 q8 会逐 shard 读取，耗时比后续缓存命中更长；
- Transformer 和 CLIP 先用 `quantize=8`，VAE 用 `0`；内存仍不足时可尝试 q4，但质量与
  性能需要自行评估；
- 从 1024×1024、batch 1 开始，不要一开始直接使用官方 2048 示例；
- 参考图越多、分辨率越高，Qwen3-VL 条件 token 和 DiT prefix 越长；多图编辑应逐张增加；
- `compile` 保持关闭；
- 本项目的 32×32 q8 端到端开发冒烟测试峰值约 15.3 GB，但这**不是** 1024×1024 的
  内存需求；正常分辨率会使用更多激活内存，请按机器容量逐步测试。

## 9. 常见问题

### Loader 显示 `<无可用权重>`

确认模型不在插件目录或 ComfyUI 的 `models/checkpoints`，而在共享目录：

```text
/Users/apple/ComfyUI-Shared/models/mlx/<组件>/Qwen-Image-2.1
```

确认四个组件链接存在且没有断开，然后完整重启 ComfyUI。

### tokenizer 加载失败

本地目录应为：

```text
/Users/apple/ComfyUI-Shared/models/mlx/tokenizer/Qwen-Image-2.1 -> <官方 snapshot>/processor
```

不是指向 snapshot 根目录，也不是指向不存在的 `<snapshot>/tokenizer`。

### 报“模型大类不匹配”

三个 Loader 必须全部选择 `qwen_image_21`。不要混用：

- `qwen_image`：旧 Qwen-Image 2512 文生图；
- `qwen_edit`：旧 Qwen-Image-Edit 2511；
- `qwen_image_21`：本指南使用的 2.1 原生统一生成/编辑模型。

### 编辑时报“没有连接同一个 ref_images”

不要创建三套 VAE 编码节点。把**同一个** `MlxVAEEncoder.ref_images` 输出分叉连接到正向
文本编码器、负向文本编码器和采样器。

### 报 `max_length` 不够

Qwen-Image 2.1 的 `text_config.max_position_embeddings` 是 **262144**。本仓库的
`MLX 条件加载器.max_length` 默认值和最大值都已设为 262144，T2I、单图编辑和多图编辑
工作流也使用这个值，不会再用 512/4096 这种过小值限制 PE 重写后的提示词。其他模型如
有需要仍可手动把该值调小。编辑模式中参考图的视觉 token 也占用这个总上下文窗口；如果
同时连接很多高分辨率参考图仍然超限，需要减少参考图或降低参考图尺寸，而不是继续增大
超过模型上限的数值。

### 报缺少权重、shape 不符或只下载了几 MB

这通常表示下载不完整，或拿到的是 Git LFS/Xet 指针而不是真实 safetensors。重新执行
`hf download`，并确认完整目录约 31 GB。不要只手工下载配置文件。

## 10. 快速核对清单

- [ ] 官方 `Qwen/Qwen-Image-2.1` 完整下载约 31 GB；
- [ ] `transformer`、`text_encoder`、`processor`、`vae` 均完整；
- [ ] 四个组件分别链接到 `/Users/apple/ComfyUI-Shared/models/mlx` 对应目录；
- [ ] 本地 `tokenizer` 链接指向官方 `processor`；
- [ ] 三个 Loader 都是 `qwen_image_21 + Qwen-Image-2.1`；
- [ ] Transformer/CLIP q8，VAE q0，compile 关闭；
- [ ] 编辑时同一个 `ref_images` 连接正向、负向和采样器；
- [ ] 放置模型后已完整重启 ComfyUI。