# ComfyUI-MLX-GEN 节点与工作流使用手册

本文档说明如何安装和使用本仓库的 MLX 节点、模型应该放到哪些目录、如何导入
`workflows/` 下的示例工作流，以及不同模型家族的正确连线方式。

如果只需要安装和使用 Qwen-Image 2.1，请直接阅读：
**[Qwen-Image 2.1 下载、安装与使用指南](QWEN_IMAGE_21_USAGE_ZH.md)**。

> 适用平台：Apple Silicon（M 系列芯片）。MLX 不能在 Intel Mac、Windows 或普通 CUDA
> 环境中运行。

## 1. 快速开始

### 1.1 安装节点

把仓库放到 ComfyUI 的 `custom_nodes` 目录。可以直接克隆，也可以把当前开发目录软链接进去：

```bash
# 方案一：直接克隆
cd /你的/ComfyUI/custom_nodes
git clone <本仓库地址> ComfyUI-MLX-GEN

# 方案二：软链接当前仓库
ln -s /你的/ComfyUI-MLX-GEN源码目录 \
  /你的/ComfyUI/custom_nodes/ComfyUI-MLX-GEN
```

使用 **ComfyUI 实际运行所用的 Python** 安装依赖：

```bash
cd /你的/ComfyUI/custom_nodes/ComfyUI-MLX-GEN
/你的/ComfyUI/.venv/bin/python -m pip install -r requirements.txt
```

如果 ComfyUI 不是虚拟环境安装，把命令开头换成启动 ComfyUI 时实际使用的 Python。
安装完成后重启 ComfyUI。

依赖版本以 `requirements.txt` 为准，当前主要包括：

- `mlx>=0.32.0,<0.33.0`
- `mflux==0.19.1`
- `mlx-audio==0.5.1`
- `mlx-whisper==0.4.3`
- `tiktoken>=0.9`

> 不要同时安装 `mlx-gen` 与本项目固定的 `mflux==0.19.1`。两个发行包都会提供名为
> `mflux` 的 Python 模块，但 MLX 版本约束不同，容易造成导入错误或运行时不兼容。

需要清理旧环境时，可以在 ComfyUI 的 Python 环境中执行：

```bash
python -m pip uninstall -y mlx-gen mflux
python -m pip install -r /你的/ComfyUI/custom_nodes/ComfyUI-MLX-GEN/requirements.txt
python -m pip show mlx mflux
```

### 1.2 创建模型目录

**当前代码没有使用 ComfyUI 默认的 `models/checkpoints`、`models/vae` 等目录。** 插件使用
相对于插件根目录的模型目录：

```text
<ComfyUI-MLX-GEN>/models/mlx
```

先创建所需子目录：

```bash
cd /你的/ComfyUI/custom_nodes/ComfyUI-MLX-GEN
mkdir -p models/mlx/{transformer,unconditional_transformer,vae,audio_vae,text_encoder,tokenizer,lora}
```

`MODEL_ROOT` 由 `paths.py` 所在位置解析为 `<插件根目录>/models/mlx`，与 macOS 用户名和
启动 ComfyUI 时的工作目录无关，不需要再修改源码。希望把大模型放到其他磁盘时，请保留
这里的组件目录，并在里面建立指向实际权重的绝对软链接。

推荐的总目录结构如下：

```text
<ComfyUI-MLX-GEN>/models/mlx/
├── transformer/               # 扩散 Transformer、TTS 主模型、YuE2 主模型、Whisper
├── unconditional_transformer/ # Ideogram 4 的无条件 Transformer
├── vae/                       # 图片/视频 VAE、YuE2 VAE
├── audio_vae/                 # MiniMax-H3 音频 VAE
├── text_encoder/              # 文本编码器
├── tokenizer/                 # tokenizer
└── lora/                      # .safetensors LoRA
```

普通图片模型通常要把同一套 checkpoint 的组件分别放入四个同名目录：

```text
mlx/
├── transformer/<权重集名称>/
├── text_encoder/<权重集名称>/
├── tokenizer/<权重集名称>/
└── vae/<权重集名称>/
```

组件可以是实际目录，也可以是指向真实模型目录的绝对软链接。模型很大时建议使用软链接，
不需要复制多份权重。

### 1.3 导入示例工作流

1. 启动 ComfyUI。
2. 把 `workflows/*.json` 拖入 ComfyUI 画布，或从菜单选择 **Load/打开工作流**。
3. 检查工作流里的 `MLX CLIP 加载`、`MLX 模型加载`、`MLX VAE 加载`：
   - `model_type` 必须属于同一个模型家族；
   - 三个节点应选择同一套权重的对应组件；
   - 如果示例中的目录名与你本地创建的软链接名不同，请重新选择。
4. 修改提示词、输入图片或输入音频。
5. 点击 **Queue Prompt/执行**。

模型下拉框是在扫描模型目录后生成的。新放入模型后如果下拉框没有更新，请完整重启
ComfyUI，而不只是刷新浏览器。

## 2. 模型放置总表

下表中的“目录名”与示例工作流序列化的默认值一致。模型仓库只是常见来源；下载后仍需按
组件拆分或建立软链接，插件不会在路径错误时自动联网下载。

| 模型用途 | `model_type` | 示例目录名 | 需要的组件目录 | 常见模型来源 |
| --- | --- | --- | --- | --- |
| Z-Image Turbo 文生图 | `z_image` | `z-image-turbo-8bit` | `transformer`、`text_encoder`、`tokenizer`、`vae` | `AbstractFramework/z-image-turbo-8bit` |
| FLUX.2 Klein 文生图/编辑 | `flux2` | `flux.2-klein-9b-8bit` | `transformer`、`text_encoder`、`tokenizer`、`vae` | `AbstractFramework/flux.2-klein-9b-8bit` |
| Qwen-Image 2.1 统一生成/编辑 | `qwen_image_21` | `Qwen-Image-2.1` | `transformer`、`text_encoder`、`tokenizer`、`vae` | `Qwen/Qwen-Image-2.1` |
| Qwen-Image 2512 文生图 | `qwen_image` | `qwen-image-2512-8bit` | `transformer`、`text_encoder`、`tokenizer`、`vae` | `AbstractFramework/qwen-image-2512-8bit` |
| Qwen-Image-Edit 2511 | `qwen_edit` | `qwen-image-edit-2511-8bit` | `transformer`、`text_encoder`、`tokenizer`、`vae` | `AbstractFramework/qwen-image-edit-2511-8bit` |
| Ideogram 4 FP8 | `ideogram4` | `ideogram-4-fp8` | 上述四类，再加 `unconditional_transformer` | `ideogram-ai/ideogram-4-fp8` |
| MiniMax-H3 视频/音频 | `minimax_h3` | `MiniMax-H3` | `transformer`、`text_encoder`、`tokenizer`、`vae`、`audio_vae` | `MiniMaxAI/MiniMax-H3` |
| MiniMax-H3 参考生视频（Ref2VA） | `minimax_h3` | `MiniMax-H3-REF`（本地 `MiniMax-H3-ref`） | 只替换 `transformer`（snapshot 的 `transformer_ref`）；其他组件仍取 `MiniMax-H3` | `MiniMaxAI/MiniMax-H3` |
| MiniMax-H3 预量化 Transformer | `minimax_h3` | `MiniMax-H3-MLX-8bit` | 只替换 `transformer`；其他组件仍取 `MiniMax-H3` | `pipenetwork/MiniMax-H3-MLX-8bit` |
| YuE2-3B 音乐 | `yue2` | `YuE2-3B-MLX-4bit` | 完整变体目录分别放入 `transformer`、`vae` | `npario/YuE2-3B-MLX` |
| Breeze-TTS-2 | `breeze_tts2` | `Breeze-TTS-2-mlx-4bit` | 完整 checkpoint 只放 `transformer` | `mlx-community/Breeze-TTS-2-mlx-4bit` |
| Whisper ASR | 不经过通用 Loader | `whisper-large-v3-mlx` | 完整 checkpoint 放 `transformer` | `mlx-community/whisper-large-v3-mlx` |

### 2.1 普通图片模型：Z-Image、FLUX.2、Qwen

这四类模型使用相同的组件布局。以 Z-Image 为例：

```text
<ComfyUI-MLX-GEN>/models/mlx/
├── transformer/z-image-turbo-8bit/
├── text_encoder/z-image-turbo-8bit/
├── tokenizer/z-image-turbo-8bit/
└── vae/z-image-turbo-8bit/
```

如果 Hugging Face snapshot 本身是 diffusers/mflux 风格目录，可以这样建立软链接：

```bash
MLX_ROOT=/你的/ComfyUI/custom_nodes/ComfyUI-MLX-GEN/models/mlx
SNAPSHOT=/模型实际位置/z-image-turbo-8bit
MODEL_NAME=z-image-turbo-8bit

ln -s "$SNAPSHOT/transformer"  "$MLX_ROOT/transformer/$MODEL_NAME"
ln -s "$SNAPSHOT/text_encoder" "$MLX_ROOT/text_encoder/$MODEL_NAME"
ln -s "$SNAPSHOT/tokenizer"    "$MLX_ROOT/tokenizer/$MODEL_NAME"
ln -s "$SNAPSHOT/vae"          "$MLX_ROOT/vae/$MODEL_NAME"
```

FLUX.2、Qwen-Image 和 Qwen-Image-Edit 只需替换 `SNAPSHOT` 与 `MODEL_NAME`：

```text
flux.2-klein-9b-8bit
qwen-image-2512-8bit
qwen-image-edit-2511-8bit
```

Qwen-Image 2.1 的官方仓库使用 `processor/` 而不是 `tokenizer/`，不能直接照抄上面的第三条
链接；下载与正确的四组件链接命令见
[Qwen-Image 2.1 专项指南](QWEN_IMAGE_21_USAGE_ZH.md#4-把模型放到插件能扫描的位置)。

注意：

- 同一条工作流里的 Transformer、CLIP、VAE 不要混用不同模型目录。
- `qwen_image` 是纯文生图模型，不能连接参考图。
- `qwen_edit` 是参考图编辑模型，必须连接参考图，并使用 `MlxQwenEditEncoder`。
- `quantize` 通常按 checkpoint 的实际量化档位选择，例如 8-bit 权重选 `8`。

### 2.2 Ideogram 4 FP8

Ideogram 4 需要条件与无条件两套 Transformer：

```text
<ComfyUI-MLX-GEN>/models/mlx/
├── transformer/ideogram-4-fp8/
├── unconditional_transformer/ideogram-4-fp8/
├── text_encoder/ideogram-4-fp8/
├── tokenizer/ideogram-4-fp8/
└── vae/ideogram-4-fp8/
```

示例软链接：

```bash
MLX_ROOT=/你的/ComfyUI/custom_nodes/ComfyUI-MLX-GEN/models/mlx
SNAPSHOT=/模型实际位置/ideogram-4-fp8
MODEL_NAME=ideogram-4-fp8

ln -s "$SNAPSHOT/transformer"               "$MLX_ROOT/transformer/$MODEL_NAME"
ln -s "$SNAPSHOT/unconditional_transformer" "$MLX_ROOT/unconditional_transformer/$MODEL_NAME"
ln -s "$SNAPSHOT/text_encoder"              "$MLX_ROOT/text_encoder/$MODEL_NAME"
ln -s "$SNAPSHOT/tokenizer"                 "$MLX_ROOT/tokenizer/$MODEL_NAME"
ln -s "$SNAPSHOT/vae"                       "$MLX_ROOT/vae/$MODEL_NAME"
```

如果 `transformer/ideogram-4-fp8` 指向 snapshot 下的 `transformer/`，并且真实目录旁边存在
`unconditional_transformer/`，插件也会尝试自动寻找同级目录。但显式建立五个链接最直观，
也最容易排查问题。

推荐设置：

- Transformer、CLIP、VAE 的 `quantize` 都选 `0`，保留 checkpoint 原生 FP8/BF16；
- 1024×1024、batch 1；
- `scheduler=ideogram4_default`；也可以选择 `ideogram4_quality` 或
  `ideogram4_turbo`；
- 负向文本会被忽略，但为保持通用采样器连线，仍需连接负向条件节点；
- 目前只支持文生图，不支持 Remix、蒙版或参考图编辑。

Ideogram 4 是 gated 模型，需先在 Hugging Face 接受模型许可。

### 2.3 MiniMax-H3

MiniMax-H3 的标准组件布局（`transformer` 有 Base 与 REF 两条，其余组件共用）：

```text
<ComfyUI-MLX-GEN>/models/mlx/
├── transformer/MiniMax-H3/       # snapshot/transformer（Base：t2v / 首尾帧）
├── transformer/MiniMax-H3-ref/   # snapshot/transformer_ref（Ref2VA：参考生视频）
├── text_encoder/MiniMax-H3/      # snapshot/text_encoder_back
├── tokenizer/MiniMax-H3/         # snapshot/tokenizer
├── vae/MiniMax-H3/               # snapshot/vae，视频 VAE
└── audio_vae/MiniMax-H3/         # snapshot/audio_vae，音频 VAE
```

特别注意：文本编码器应指向官方 checkpoint 的 **`text_encoder_back`**，不是键名不同的
社区 NVFP4 `text_encoder` 单文件。

```bash
MLX_ROOT=/你的/ComfyUI/custom_nodes/ComfyUI-MLX-GEN/models/mlx
SNAPSHOT=/模型实际位置/MiniMax-H3

ln -s "$SNAPSHOT/transformer"       "$MLX_ROOT/transformer/MiniMax-H3"
ln -s "$SNAPSHOT/transformer_ref"   "$MLX_ROOT/transformer/MiniMax-H3-ref"
ln -s "$SNAPSHOT/text_encoder_back" "$MLX_ROOT/text_encoder/MiniMax-H3"
ln -s "$SNAPSHOT/tokenizer"         "$MLX_ROOT/tokenizer/MiniMax-H3"
ln -s "$SNAPSHOT/vae"               "$MLX_ROOT/vae/MiniMax-H3"
ln -s "$SNAPSHOT/audio_vae"         "$MLX_ROOT/audio_vae/MiniMax-H3"
```

**参考生视频（多图 / 纯参考）必须把 `MlxTransformerLoader` 选成
`MiniMax-H3-ref`**：Base 的 `transformer` 只支持 0~2 张图、只认首 / 尾两个
锚点槽，`transformer_ref`（H3-Base-Ref2VA）才支持最多 9 张参考图；
首 / 尾帧与视频续写仍用 Base。两份 transformer 的结构与 `config.json` 相同，
差的只是训练权重，因此换 checkpoint 不需要改代码——但选错了采样器会直接报错。

每个 H3 组件目录都必须保留自己的 `config.json` 和 `.safetensors` 权重；这包括
Transformer、`text_encoder_back`、视频 VAE 与音频 VAE。当前实现会分别调用严格的
配置加载器，任一组件缺少 `config.json` 都会直接报错，没有内置默认配置回退。

二阶段放大（可选）：把放大模型放进 `upscaler/` 目录即可，节点下拉会自动列出（该目录
**不需要**登记到 `paths.COMPONENT_DIRS`，节点按目录名扫盘）：

```bash
ln -s /路径/minimax_h3_latent_upscaler_3d_bf16.safetensors \
  /你的/ComfyUI/custom_nodes/ComfyUI-MLX-GEN/models/mlx/upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors
```

文件名里带 `3d` 的是空间+时间联合处理的变体（本插件移植的就是它，全上下文一次前向
`640×352 → 1280×704` 实测约 4–7 秒）。用法见 §4.5 与 §5.4。

#### 使用 PipeNetwork 8-bit Transformer

`pipenetwork/MiniMax-H3-MLX-8bit` 只替换 Transformer，文本编码器、tokenizer、视频 VAE
和音频 VAE 仍使用上面的官方 `MiniMax-H3` 组件：

```bash
ln -s /HuggingFace缓存/models--pipenetwork--MiniMax-H3-MLX-8bit \
  /你的/ComfyUI/custom_nodes/ComfyUI-MLX-GEN/models/mlx/transformer/MiniMax-H3-MLX-8bit
```

该链接可以指向 Hugging Face cache 外壳；插件会通过 `refs/main` 定位
`snapshots/<revision>`。也可以直接链接到具体 snapshot。

导入 H3 工作流后：

- `MlxTransformerLoader`：路径选 `MiniMax-H3-MLX-8bit`；
- `MlxClipLoader`：路径仍选 `MiniMax-H3`；
- 视频 VAE 与音频 VAE：路径仍选 `MiniMax-H3`；
- Transformer 与文本编码器的 `quantize` 选 `8` 或 `4`，不能选 `0`；
- `compile` 会自动关闭；
- 从示例的 640×352、124 帧、50 步开始验证。H3 对统一内存要求很高。

H3 帧数必须满足 `17n+5`。当前节点下拉框按代码中的 `valid_frame_counts()` 生成，共有
`5, 22, 39, …, 345` 这 21 个合法值；**124 只是示例工作流使用的值，不是协议下限**。
宽高必须为 32 的倍数，宽高比必须在 1:4 到 4:1 之间。示例固定使用 24 fps。

### 2.4 YuE2-3B

从 `npario/YuE2-3B-MLX` 下载后，选择一个完整精度变体目录，例如 `4bit/`。目录中应包含：

```text
config.json
model.safetensors
qwen.tiktoken
yue2_generation_config.json
vae_config.json
vae.safetensors
```

把 **同一个完整变体目录** 分别链接到 `transformer/` 与 `vae/`：

```bash
MLX_ROOT=/你的/ComfyUI/custom_nodes/ComfyUI-MLX-GEN/models/mlx
VARIANT=/模型实际位置/YuE2-3B-MLX/4bit

ln -s "$VARIANT" "$MLX_ROOT/transformer/YuE2-3B-MLX-4bit"
ln -s "$VARIANT" "$MLX_ROOT/vae/YuE2-3B-MLX-4bit"
```

YuE2 没有需要单独放置的文本编码器；`MlxClipLoader` 会把名称包含 `yue2` 的 Transformer
目录也加入候选列表。三个 Loader 都选择 `model_type=yue2`，路径选择同一个变体名称。

插件也兼容指向 `model.safetensors` 和 `vae.safetensors` 的单文件软链接，但目录软链接更不易
出错，因为插件还需要从同级目录读取配置、tokenizer 和另一组权重。

### 2.5 Breeze-TTS-2

Breeze checkpoint 已经包含主模型、文本编码器与 audio tokenizer。完整 checkpoint 只需放入
`transformer/`，不需要向 `vae/`、`text_encoder/`、`tokenizer/` 再复制：

```text
<ComfyUI-MLX-GEN>/models/mlx/
└── transformer/Breeze-TTS-2-mlx-4bit/  # 完整 Hugging Face snapshot
```

```bash
ln -s /模型实际位置/Breeze-TTS-2-mlx-4bit \
  /你的/ComfyUI/custom_nodes/ComfyUI-MLX-GEN/models/mlx/transformer/Breeze-TTS-2-mlx-4bit
```

`MlxTransformerLoader` 和 `MlxVAELoader` 都选择 `model_type=breeze_tts2` 和同一个 checkpoint
名称。VAE Loader 在 `vae/` 找不到时会回退到 `transformer/`；它只是传递兼容句柄，不会
重复加载一份模型。

示例工作流分别使用 `Breeze-TTS-2-mlx-4bit` 与 `Breeze-TTS-2-mlx`。如果你只下载了其中
一种，导入工作流后同时修改 Transformer Loader 和 VAE Loader 的路径即可。

### 2.6 MLX Whisper

Whisper 完整 checkpoint 放到：

```text
<ComfyUI-MLX-GEN>/models/mlx/transformer/whisper-large-v3-mlx/
```

该目录必须满足：

- 存在 `config.json`，且其中 `model_type` 为 `whisper`；
- 存在 `weights.npz` 或 `weights.safetensors`。

`MlxWhisperTranscribe` 只显示满足上述条件的本地目录，不会因为路径错误而静默联网下载。

## 3. 示例工作流一览

| 工作流文件 | 用途 | 默认模型/关键参数 |
| --- | --- | --- |
| `workflows/z-image-turbo-512.json` | Z-Image 512×512 文生图 | 4 步、`linear`、guidance 1.0 |
| `workflows/z-image-turbo-768.json` | Z-Image 768×768 文生图 | 6 步、`linear`、guidance 1.0 |
| `workflows/flux2-klein-9b-512.json` | FLUX.2 Klein 512×512 文生图 | 4 步、flow-match、guidance 1.0 |
| `workflows/flux2-klein-9b-1024.json` | FLUX.2 Klein 1024×1024 文生图 | 4 步、flow-match、guidance 1.0 |
| `workflows/flux2-klein-9b-edit.json` | FLUX.2 单图编辑 | 参考图 VAE 编码后接 `ref_images` |
| `workflows/flux2-klein-9b-edit-multi.json` | FLUX.2 异尺寸多图编辑 | 使用 `MlxRefImageSet` 保留顺序和原始尺寸 |
| `workflows/qwen-image-2512.json` | Qwen-Image 2512 文生图 | 20 步、flow-match、guidance 4.0 |
| `workflows/qwen-image-edit-multi.json` | Qwen-Image-Edit 多图编辑 | 20 步、`linear`、guidance 2.5 |
| `workflows/qwen-image-2.1.json` | Qwen-Image 2.1 文生图 | 40 步、flow-match、guidance 1.0、RGBA |
| `workflows/qwen-image-2.1-edit.json` | Qwen-Image 2.1 原生图片编辑 | 同一 `ref_images` 接正向、负向与采样器，block-causal prefix KV |
| `workflows/ideogram-4-fp8.json` | Ideogram 4 FP8 文生图 | 1024×1024、`ideogram4_default` |
| `workflows/minimax-h3-t2v-no-audio.json` | MiniMax-H3 文生视频 | 640×352、124 帧、50 步、无音轨 |
| `workflows/minimax-h3-t2va.json` | MiniMax-H3 文生视频和立体声 | 在视频工作流上增加 `audio_vae` |
| `workflows/minimax-h3-i2va-first-frame.json` | MiniMax-H3 首帧生视频（Base） | 1 张图钉成第一帧 |
| `workflows/minimax-h3-i2va-last-frame.json` | MiniMax-H3 尾帧生视频（Base） | 1 张图钉成最后一帧 |
| `workflows/minimax-h3-i2va-first-last-frame.json` | MiniMax-H3 首尾帧生视频（Base） | 2 张图分别钉首 / 尾帧 |
| `workflows/minimax-h3-single-image-to-video.json` | MiniMax-H3 单图生视频（Base） | 生成器版本，1 张图钉首帧 |
| `workflows/minimax-h3-multi-image-to-video.json` | MiniMax-H3 多图参考生视频（**H3-REF**） | 3 张参考图，第 1 张钉首帧 |
| `workflows/minimax-h3-reference-only-to-video.json` | MiniMax-H3 纯参考生视频（**H3-REF**） | 3 张只进 presentation，不钉锚点 |
| `workflows/minimax-h3-all-reference-to-video.json` | MiniMax-H3 全能参考生视频（**H3-REF**） | 4 张只进 presentation + Ref2VA 的 4 步加速 LoRA |
| `workflows/minimax-h3-video-continuation-keep-audio.json` | 源视频续写并拼成片（Base） | `GetVideoComponents` + `ConcatenateVideo` + `AudioConcat` |
| `workflows/minimax-h3-video-continuation-drop-audio.json` | 只出新片段（Base） | 丢弃原声，用 H3 生成的音轨 |
| `workflows/minimax-h3-video-continuation-replace-audio.json` | 只出新片段 + 外部配乐（Base） | `LoadAudio` 整段替换音轨 |
| `workflows/minimax-h3-two-stage-upscale.json` | MiniMax-H3 二阶段：低分采样 → latent 放大 → 高分精修 | 640×352 ×2 → 1280×704、8 步停第 4 步、精修 4 步 |
| `workflows/minimax-h3-two-stage-upscale-lora.json` | 同上 + 8 步加速 LoRA（两段共用同一条 model 线） | 4 + 4 = 8 步、切点 σ=0.9231、`strength=1.0` |
| `workflows/yue2-3b.json` | YuE2 风格+歌词生成音乐 | 32 步、最长先设约 8 秒 |
| `workflows/breeze-tts2.json` | Breeze 内置说话人 TTS | `speaker` 模式、S0、24 kHz 单声道 |
| `workflows/breeze-tts2-voice-clone.json` | Breeze 手工逐字稿声音克隆 | 参考音频 + 完全一致的逐字稿 |
| `workflows/breeze-tts2-voice-clone-asr.json` | Breeze + Whisper 自动转写克隆 | Whisper 输出直接接 `ref_text` |

## 4. 常用工作流连线

### 4.1 普通文生图

适用于 `z_image`、`flux2`、`qwen_image`、`qwen_image_21` 和 `ideogram4`：

```text
MlxClipLoader
  ├─→ MlxTextEncoder（正向提示词）─→ positive
  └─→ MlxTextEncoder（负向提示词）─→ negative

MlxTransformerLoader ─→ MlxKSamplerMLX.model
positive / negative ──→ MlxKSamplerMLX
MlxKSamplerMLX.latents ─→ MlxVAEDecoder 或 MlxVAEDecodeRawPIL
MlxVAEDecoder.images ───→ MlxPilToTorch ─→ PreviewImage / SaveImage
```

推荐新工作流使用 `MlxVAELoader + MlxVAEDecoder`。`MlxVAEDecodeRawPIL` 是保留给旧工作流的
一体式解码节点，仍可正常使用。

不同家族的关键采样参数：

| 家族 | steps | guidance | scheduler | compile |
| --- | ---: | ---: | --- | --- |
| Z-Image Turbo | 示例 4 或 6 | 1.0 | `linear` | 可开 |
| FLUX.2 Klein | 4 | 1.0 | `flow_match_euler_discrete` | 可开 |
| Qwen-Image 2512 | 20 | 4.0 | `flow_match_euler_discrete` | 关闭 |
| Qwen-Image 2.1 | 40 | 1.0 | `flow_match_euler_discrete` | 关闭 |
| Ideogram 4 | 由预设决定 | 由预设决定 | `ideogram4_default/quality/turbo` | 可开 |

### 4.2 FLUX.2 参考图编辑

单图或同尺寸批次：

```text
Load Image / Batch Images ─→ MlxVAEEncoder.images
MlxVAELoader ──────────────→ MlxVAEEncoder.vae
MlxVAEEncoder.ref_images ──→ MlxKSamplerMLX.ref_images
```

### 4.3 Qwen-Image 2.1 原生图片编辑

不要使用 legacy 的 `MlxQwenEditEncoder`。三个加载器都选 `qwen_image_21` +
`Qwen-Image-2.1`，并将同一个参考图 handle 显式送到三处：

```text
Load Image / MlxRefImageSet ─→ MlxVAEEncoder
MlxVAEEncoder.ref_images ────┬→ MlxTextEncoder（正向）.ref_images
                              ├→ MlxTextEncoder（负向）.ref_images
                              └→ MlxKSamplerMLX.ref_images
```

2.1 的参考条件包含两部分：Qwen3-VL 视觉特征和归一化 64 通道 VAE latent。插件用
同一份保持比例、对齐 32 倍数的像素生成二者，并在 DiT 中按视觉槽插入参考 latent。
支持最多 10 张图；异尺寸多图通过 `MlxRefImageSet` 保留顺序与各自宽高。

多张图片尺寸不同时，不要使用 ComfyUI 的 `Batch Images`，因为它会把后面的图片缩放/裁切为
第一张的尺寸。应使用：

```text
Load Image 1 ─┐
Load Image 2 ─┼→ MlxRefImageSet.ref_source ─→ MlxVAEEncoder.ref_source
Load Image 3 ─┘
```

`image1 → image2 → image3` 的顺序就是“第一张、第二张、第三张”的语义顺序。FLUX.2 的文本
条件仍使用普通 `MlxTextEncoder`；参考图只进入 VAE 编码链路。

### 4.4 Qwen-Image-Edit 多图编辑

Qwen 编辑与 FLUX.2 编辑不同：参考图既要送进 VAE，也要送进正、负两个视觉文本条件节点：

```text
Load Image 1 ─┐
Load Image 2 ─┴→ Batch Images ─┬→ MlxVAEEncoder.images
                               ├→ MlxQwenEditEncoder（正向）.images
                               └→ MlxQwenEditEncoder（负向）.images

MlxVAEEncoder.ref_images ─────────→ MlxKSamplerMLX.ref_images
两个 MlxQwenEditEncoder.condition ─→ positive / negative
```

必须遵守：

- 三个 Loader 都选 `qwen_edit`，不能选 `qwen_image`；
- 正向和负向 `MlxQwenEditEncoder` 必须接同一批图；
- 负向文本可以留空，但节点和图片连接不能省略；
- VAE 编码后的 `width`、`height` 必须与采样器完全一致。最稳妥的方法是把采样器的宽高
  widget 转成输入，并连接 `MlxVAEEncoder.width/height`；
- Qwen 编辑当前使用同尺寸批次入口，不使用 `MlxRefImageSet.ref_source`。

### 4.5 MiniMax-H3 视频和音频

仅视频：

```text
MlxClipLoader → MlxTextEncoder ─┬→ MlxKSamplerMLX.positive
                                └→ MlxKSamplerMLX.negative（H3 会忽略负向语义）
MlxTransformerLoader ────────────→ MlxKSamplerMLX.model
MlxKSamplerMLX.latents ──────────→ MlxVAEDecodeRawPIL
MlxVAEDecodeRawPIL.images ───────→ MlxPilToTorch.images
MlxPilToTorch.images ────────────→ CreateVideo.images → SaveVideo
```

视频加音频时，再增加一个 `MlxVAELoader`：

```text
MlxVAELoader(model_type=minimax_h3, role=audio_vae)
  ─→ MlxVAEDecodeRawPIL.audio_vae

MlxPilToTorch.audio ─→ CreateVideo.audio
CreateVideo fps = 24
```

H3 是 guidance 蒸馏模型，负向文本、guidance 和通用 scheduler widget 不参与 H3 的实际去噪
语义；请保留示例连线和默认值。真正影响结果和计算量的是提示词、seed、steps、分辨率、
`num_frames`、`video_shift` 与 `audio_shift`。

### 4.6 MiniMax-H3 二阶段放大（低分 → latent 放大 → 高分精修）

把「高分辨率直出」拆成两段：先在低分辨率跑完 σ 网格的前半段，用 3D 放大网络在
**latent 空间**放大（不经过 VAE），再在目标分辨率上接着跑剩下的 σ。默认 8 步切在
第 4 步时，高分辨率只需要跑 4 步，所以比「直接 8 步高分辨率出片」快 —— 按打包行数算，
`1280×704` 的单步成本是 `640×352` 的 **3.79 倍**。

```text
MlxClipLoader → MlxTextEncoder ─┬→ MlxH3FirstPassSampler.positive
                                └→ MlxH3FirstPassSampler.negative
MlxTransformerLoader ────────────→ MlxH3FirstPassSampler.model
MlxH3FirstPassSampler(steps=8, width=640, height=352, num_frames=124, stop_at_step=4)
  → MlxH3LatentUpscaler(scale=2.0)        # 目标画布 = 上一段 × 倍率（自动吸附 32 倍数）
  → MlxH3SecondPassSampler(steps=4, start_at_sigma=0.7, audio_mode=follow_video)
  → MlxVAEDecoder(vae + audio_vae) → MlxPilToTorch → CreateVideo → SaveVideo
```

三个新节点与既有节点**完全并存**：老工作流一行都不用改。新链路也可以只取其中一段
（例如只接 `MlxH3FirstPassSampler` 并把 `stop_at_step` 设为 0：结果与既有
「MLX 采样器」逐位一致）。

参数要点：

- `stop_at_step=4` 指的是**母网格的第 4 个点**（σ_video 0.9231 / σ_audio 0.75），
  不是「4 步调度」—— 母网格必须保留，否则二阶段接不上；`0` = 跑满 `steps`；
- **`output_mode` 必须保持 `denoised (x0)`**（除非你只用一阶段直出、不接放大）：
  它交给下游的是「最后一个工作 σ 处的一步 x0 估计」，而放大网络是在**干净 latent 对**上
  训练的 —— 把 92% 是噪声的 `latent (x_t)` 交给它，它会把噪声一起放大，二阶段重采样后
  画面就是**混沌**的（ComfyUI 参考工作流也是取 `SamplerCustomAdvanced.denoised_output`
  这一路进放大节点）。`x0` 复用最后一步已经算好的模型输出，**不额外增加前向**；
- 放大节点**没有长宽输入**：目标画布 = 上一段画布 × `scale`，按 32 的倍数吸附
  （平局取上，只放大不缩小）：`640×352 ×2 → 1280×704`、`×1.5 → 960×544`、`×2.5 → 1600×896`；
- 示例工作流的 `model_name` 默认选
  **`minimax_h3_latent_upscaler_3d_bf16.safetensors`**（完整文件名，不能写成裸名
  `minimax_h3_latent_upscaler_3d`）。它必须位于 `models/mlx/upscaler/`；节点启动时会
  扫描该目录并按文件名解析。放大器进出都是 **H3 normalized latent**：一阶段
  `video_rows` 已经是 `(z - mean) / std`，VAE decode 才做反归一化；不要在放大节点
  前后再次添加 VAE mean/std。放大器会检查输出是否有限、shape 是否正确以及幅度是否
  失控；异常 checkpoint 会在进入 VAE 前直接报错，而不是生成雪花视频。细节与复现步骤见
  [`docs/MINIMAX_H3_VIDEO_IMPLEMENTATION.md`](docs/MINIMAX_H3_VIDEO_IMPLEMENTATION.md) 附录 F；
- `start_at_sigma`：`0.7`（默认，保留较多已放大结构）；填 `0.9231` 与 ComfyUI 的
  `SplitSigmas(step=4)` 语义等价（重绘更多）；越小越接近「只锐化」；
- `audio_mode`：`follow_video`（默认，与 CUDA 参考一致）/ `continue_stage1`（最保音频）/
  `regenerate`（重画音频，风险大）。音频不放大、不重编码，只跟着视频一起收尾；
- 二阶段会**重新加载一次 transformer**（阶段一结束就释放了），页缓存热时约 16 秒。

#### 挂加速 LoRA（`minimax-h3-two-stage-upscale-lora.json`）

加速 LoRA 不是"插在模型中间"，而是**挂在 model handle 上、在权重物化时融合**
（`MlxModelLoraApply` → `LoraRef` → 加载时 `apply_transformer_loras`）。因此要点是：

- 挂在**一阶段与二阶段共用的那条 model 线**上（`transformer → LoRA → 两个采样器`）：
  只给其中一段挂，等于中途换模型，轨迹会乱；
- `strength` 保持 **1.0**（LightX2V Turbo 适配器按 1.0 发布）；
- **两段步数之和 = 适配器标称步数**，且切点要落在母网格点上（这样二阶段子网格与母网格
  后半段**逐点一致**，合计就是完整的训练轨迹，只是后半段在 2× 分辨率上跑）：

| 适配器 | 一阶段 | 二阶段 | 相对成本 | 说明 |
| --- | --- | --- | --- | --- |
| **8 步**（示例默认） | `steps=8, stop_at_step=4` | `steps=4, start_at_sigma=0.9231` | 19.2 | 与母网格 index 4→8 一致；高分有 4 步修放大痕迹 |
| 8 步（更省） | `steps=8, stop_at_step=6` | `steps=2, start_at_sigma=0.8` | 13.6 | 与母网格 index 6→8 一致；高分只 2 步 |
| 4 步 | `steps=4, stop_at_step=2` | `steps=2, start_at_sigma=0.9231` | 9.6 | 总 4 步；低分只有 2 步，底子最弱 |

（相对成本按打包行数：单步高分 = 低分 ×3.79；直出高分 8 步 = 30.3。）

**基座模型（不挂 LoRA）的档位**：H3 基座官方档是 **40~50 步**，步数不够时交出的 x0 本身还是粗糙估计，
放大只会把它放大（成片像雪花 / 噪点）。基座下最稳的用法是「**低分先跑满 → 放大 → 只走尾部几步**」：

| 档位 | 一阶段（50 步母网格） | 二阶段 | x0 权重 | 相对成本（直出高分 50 步 = 189.5） | 估时 |
| --- | --- | --- | --- | --- | --- |
| **放大即用（推荐）** | `stop_at_step=0`（跑满，σ→0） | `steps=4, start_at_sigma=0.5106` | **48.9%** | 65.2 | ≈33 min |
| 更快 | `stop_at_step=0` | `steps=2, start_at_sigma=0.3333` | 66.7% | 57.6 | ≈29 min |
| 只锐化 | `stop_at_step=0` | `steps=1, start_at_sigma=0.1967` | 80.3% | 53.8 | ≈28 min |
| 省时切半 | `stop_at_step=38`（σ 0.7912） | `steps=12, start_at_sigma=0.7912` | 20.9% | 83.5 | ≈42 min |

> 二阶段的 `start_at_sigma` 都取**母网格上的点**（`0.5106 / 0.3333 / 0.1967 / 0.7912` 都能在 50 步母网格里
> 逐位找到），这样二阶段的子网格就是母网格的原生尾部，不会出现"一步跳完"的雪花。
> 成本口径：单步低分 = 1、单步高分 = 3.79；估时按仓库文档的 640×352 / 基座 50 步 ≈ 25 min 外推（非实测）。

- 换 LoRA / 换 strength / 改步数都会**换缓存键**（不会复用旧结果）——这是正确行为；
- 适配器任务要对：`..._fl2v_...` 是首 / 尾帧（FL2VA）、`..._ref2v_...` 是参考生视频（要 REF
  transformer 与 `<Picture N>` 提示词）、`..._taomate_3step_...` 未标任务。示例用 8 步 fl2v
  适配器给文生视频换速度，属于常见混用，画质请自行验证；
- `stop_at_step=0`（跑满）再接放大 + 精修 = 超出适配器标称步数，通常表现为过锐、噪点被放大。

### 4.7 YuE2 音乐生成

```text
MlxClipLoader
  ├→ MlxTextEncoder positive：style（曲风、乐器、BPM、情绪、唱法）
  └→ MlxTextEncoder negative：lyrics（歌词，不是负面提示词）
MlxTransformerLoader + 两条条件 → MlxKSamplerMLX
MlxVAELoader + latents → MlxVAEDecoder
MlxPilToTorch.audio → PreviewAudio / SaveAudio
```

参数说明：

- `cot=off`：不先生成 ABC 规划，速度最快；
- `cot=melody`：先规划旋律；
- `cot=full`：先规划带和弦的完整 ABC，耗时最长；
- `max_tokens` 控制语义 codec token 上限，每个 token 约 40 ms。建议先用 200（约 8 秒）
  验证工作流；
- `steps` 控制声学 latent 的 ODE 步数，不控制音频时长；
- 固定使用 `scheduler=yue2_midpoint`、`batch_size=1`；
- 输出是 48 kHz 立体声。

### 4.8 Breeze-TTS-2

推荐使用专用的 `MlxBreezeSampler`，其 `text`、`ref_text`、`instruction` 都是外部 STRING
输入，需要连接 ComfyUI 的 `PrimitiveStringMultiline` 或其他字符串节点。

通用输出链路：

```text
外部 STRING ─────────────→ MlxBreezeSampler.text
MlxTransformerLoader ────→ MlxBreezeSampler.model
MlxBreezeSampler.latents ─→ MlxVAEDecoder
MlxVAELoader ─────────────→ MlxVAEDecoder.vae
MlxVAEDecoder.images ─────→ MlxPilToTorch.audio
MlxPilToTorch.audio ──────→ PreviewAudio / SaveAudio
```

三种模式：

1. `speaker`：使用模型内置的 `S0`–`S9` 说话人，不连接参考音频。
2. `voice_clone`：
   - `Load Audio` 接 `ref_audio`；
   - `ref_text` 必须是参考音频实际内容的逐字转写；
   - 推荐 3–10 秒、背景干净、单人说话的参考音频。
3. `voice_design`：
   - 不连接 `ref_audio`；
   - 把音色、情绪、语速等描述接到 `instruction`；
   - 需要 CFG 时把 `cfg_scale` 设为非 1.0。

使用 ASR 工作流时，`MlxWhisperTranscribe.text` 会直接连接 Breeze 的 `ref_text`。普通话可选
`zh`，粤语选 `yue`，混合语言可选 `auto`。短参考音频建议使用
`temperature=0`、`condition_on_previous_text=false`。ASR 结果仍可能漏字，声音克隆质量不理想
时应先检查并手工修正逐字稿。

输出是 24 kHz 单声道音频。

## 5. 节点说明

仓库入口当前注册了 **16 个节点**。除 `MlxWhisperTranscribe` 位于 `MLX/Audio` 外，其余均在
`MLX/Gen` 分类。`model`、`CLIP`、`vae`、`condition`、`latents` 和小写 `images` 是本插件
的 handle/载荷类型，不等同于 ComfyUI 原生 Torch `MODEL`、`CLIP`、`VAE`、`LATENT`、
大写 `IMAGE`；请按示例工作流连接，不要把原生模型加载器混入 MLX 链路。

### 5.1 模型加载节点

| 节点 | 主要输入与关键参数 | 输出 | 用途与限制 |
| --- | --- | --- | --- |
| **MLX 模型加载** `MlxTransformerLoader` | `model_type`、`model_path`、`quantize`、`precision`、`compile`、`compile_cache_limit` | `model` handle | 登记 Transformer 配置，真正采样时才物化权重。目录名用于匹配模型配置。H3 的 `quantize` 只能选 4/8，且会自动关闭 `compile`；音频家族也会自动关闭 `compile`。Ideogram 4 的原生 FP8 checkpoint 应选 `quantize=0`。`compile_cache_limit` 单位是 **GB**，由 `MlxKSamplerMLX` 在物化权重前调 `mx.set_cache_limit` 真正设上（0 = 沿用 MLX 默认、不设置；改档位会改缓存键，模型会重新加载一次）。它约束的是 MLX 的 free-buffer 缓存，与 `compile` 开关无关，`compile` 关闭时同样会设。 |
| **MLX CLIP 加载** `MlxClipLoader` | `model_type`、`component`（`text_encoder`/`tokenizer`）、`path`、`precision`、`max_length`、`quantize` | `CLIP` handle | 登记条件编码器配置，编码时才加载。一般保持 `component=text_encoder`。H3 的条件编码器必须量化，`quantize` 只能选 4/8；YuE2/Breeze 没有独立编码器时可选择 `transformer/` 下的完整 checkpoint。 |
| **MLX VAE 加载** `MlxVAELoader` | `model_type`、`model_path`、`precision`、`quantize`、`role` | `vae` handle | 供编码器和解码器按需物化 VAE。同一个 handle 同时接编码与解码节点时复用实例。普通链路用 `role=vae`；H3 声音另建一个 `role=audio_vae` 的 Loader。Breeze 会从同名 `transformer/` 完整 checkpoint 回退解析。 |

三个 Loader 只返回轻量配置 handle，不代表权重已加载。相连节点的 `model_type` 必须一致；
图片模型通常还应选择同一 checkpoint 的同名 Transformer、文本编码器/tokenizer 与 VAE。

### 5.2 文本条件与 LoRA 节点

| 节点 | 主要输入 | 输出 | 用途与限制 |
| --- | --- | --- | --- |
| **MLX 文本编码器** `MlxTextEncoder` | `text`、`clip` | `condition` | 通用文本条件。图片模型通常放两个节点，分别接采样器的 `positive`/`negative`；Qwen Edit 禁止使用本节点。H3 会组装三段式 prompt 并只使用正向语义；YuE2 中正向是 style、负向是 lyrics；Breeze 推荐使用专用采样器，不走本节点。 |
| **MLX Qwen 编辑条件（带参考图）** `MlxQwenEditEncoder` | `text`、`clip`、原生 `IMAGE` 批次、`max_images`（1–8） | `condition` | 仅用于 `qwen_edit`。正负两个条件节点必须连接同一批参考图；超出 `max_images` 的尾部图片会被截断。它让 Qwen2.5-VL 同时编码文本和图片，不能用普通 `MlxTextEncoder` 替代。 |
| **MLX 模型 LoRA** `MlxModelLoraApply` | `model`、`lora`、`strength` | 新的 `model` handle | 从 `lora/` 选择文件；采样器在基础 Transformer 加载、量化完成后真正应用。支持 Z-Image、FLUX.2、Qwen Image/Edit、Ideogram 4 与 MiniMax-H3；可串联多个节点。 |
| **MLX CLIP LoRA** `MlxClipLoraApply` | `clip`、`lora`、`strength` | 新的 `CLIP` handle | 为旧工作流保留。当前没有经过验证的文本编码器 mapping；选择非空 LoRA 会明确报错，不会静默忽略。 |

`strength=0` 严格跳过该文件，等同基础模型；改变强度、顺序或组合会改变 Transformer 缓存键。
图片模型复用 `mflux==0.19.1` 的官方 mapping，并把低秩分支保留在量化基础 Linear 外层，
不会为了 LoRA 把整个 q4/q8 模型烘焙成另一种精度。Ideogram 4 的同一 LoRA 会同时应用到
conditional 与 unconditional 两套 Transformer。

### 5.3 参考图与采样节点

| 节点 | 主要输入与关键参数 | 输出 | 用途与限制 |
| --- | --- | --- | --- |
| **MLX 参考图集（多图）** `MlxRefImageSet` | 必填 `image1`，可选 `image2`–`image4`；每槽也可接一个原生 `IMAGE` 批次 | `ref_source`、图片数 `count`、文字 `report` | FLUX.2 多尺寸参考图专用。按槽位和批次顺序展开，保留各图原始尺寸；`report` 可接文本预览核对次序。随后把 `ref_source` 接到 `MlxVAEEncoder`，不要先经过会统一尺寸的原生 `Batch Images`。Qwen Edit 暂不支持该来源。 |
| **MLX VAE 编码** `MlxVAEEncoder` | `vae`，以及二选一的原生 `images` 批次或 `ref_source`；`max_reference_images`、`resize_mode`、目标 `width`/`height` | `ref_images`、实际/建议 `width`、`height` | 将参考图编码后接 `MlxKSamplerMLX.ref_images`。FLUX.2 默认 `aspect_area_crop`，可用多尺寸 `ref_source`；Qwen Edit 默认 `stretch`，只接受 `images`，并把参考图编码到目标尺寸，故输出宽高应与采样器一致。两个图片来源同时连接会报错。 |
| **MLX 采样器** `MlxKSamplerMLX` | 必填 `model`、正负 `condition`、`seed`、`steps`、`width`、`height`、`batch_size`、`guidance`、`scheduler`；可选 `ref_images`、`kv_cache`、YuE2 `cot`/`max_tokens`；H3 另用 `num_frames`/两种 shift | `latents` handle | Z-Image、FLUX.2、Qwen、Ideogram 4、H3 与 YuE2 的通用采样节点。不同家族只读取相关参数：H3 忽略负向文本、scheduler 和 guidance；YuE2 的 negative 是歌词且 batch 必须为 1；Ideogram 4 忽略负向文本；Qwen 文生图禁止参考图，Qwen Edit 必须连接参考图。编译缓存上限取自 Loader 的 `compile_cache_limit`，在加载权重前生效（它管的是 MLX 的 free-buffer 缓存，与 `compile` 开关无关，关闭编译也会设；0 = 不设置）。 |
| **MLX Breeze-TTS-2 采样器** `MlxBreezeSampler` | `model`、外部 `STRING text`、`seed`、`mode`、`speaker`、temperature/top-p/top-k、`cfg_scale`、`max_tokens`、重复惩罚；克隆/设计模式另接 `ref_audio`、外部 `ref_text` 或 `instruction` | `latents` handle | 仅用于 `breeze_tts2`。文本输入都是 socket，节点内部没有文本框；需连接 `PrimitiveStringMultiline` 等 STRING 节点。`speaker` 用内置 S0–S9；`voice_clone` 要参考音频及逐字稿；`voice_design` 要 instruction。一次生成一条语音。 |

### 5.4 MiniMax-H3 二阶段放大节点

| 节点 | 主要输入与关键参数 | 输出 | 用途与限制 |
| --- | --- | --- | --- |
| **MLX H3 一阶段采样（低分辨率）** `MlxH3FirstPassSampler` | `model`、正负 `condition`、`seed`、`steps`、`width`、`height`、`num_frames`、`video_shift`、`audio_shift`、`stop_at_step`、`output_mode` | `latents` handle | 低分辨率联合采样，可停在母网格第 k 步（`stop_at_step=0` 跑满）。`output_mode` 二选一：`denoised (x0)`（默认，最后一个工作 σ 处的一步 x0 估计 —— **接放大网络必须用它**）或 `latent (x_t)`（含噪原始 latent，跑满直出时与 `MlxKSamplerMLX` 逐位一致）。 |
| **MLX H3 Latent 放大（3D）** `MlxH3LatentUpscaler` | `latents`、`model_name`（含内置插值选项）、`scale`（1.0–4.0）、可选 `precision` | `latents` handle | 在 latent 空间放大（默认**内置三线性插值**，不加载模型；实测官方放大权重输出幅度为真值的 6.4 倍、与目标不相关，详见实现文档附录 F）。**没有长宽 widget**：目标画布 = 上一段画布 × `scale`，吸附到 32 的倍数（平局取上，只放大不缩小）。音频行原样透传；`scale=1.0` 或吸附后与输入相同时原样返回。 |
| **MLX H3 二阶段采样（放大后精修）** `MlxH3SecondPassSampler` | `model`、正负 `condition`、`latents`、`seed`、`steps`、`start_at_sigma`、`audio_mode`；可选 `keyframes` | `latents` handle | 画布 / 帧数 / 两条 shift 全取自入参 `latents`（无相关 widget）。按 `x=(1-σ₀)·x₀+σ₀·噪声` 注入后跑到 σ=0；接 `keyframes` 时锚点按新画布重新登记（源是低分图，略软化，不接则不占锚点行）。 |

三个节点只产出/转发同一个 `latents` 句柄（`kind="h3_video"`），
`(video_rows, audio_rows, plan)` 仍写进 `h3_latents` 桶，因此既有的
`MlxVAEDecoder` / `MlxPilToTorch` / `CreateVideo` / `SaveVideo` 直接能接。

### 5.5 解码、格式转换与保存节点

| 节点 | 主要输入 | 输出 | 用途与限制 |
| --- | --- | --- | --- |
| **MLX VAE 解码** `MlxVAEDecoder` | `vae`、`latents`、`batch_index`，H3 可选第二个 `audio_vae` | 插件内部小写 `images` | 推荐解码节点。图片得到 PIL 图像；H3 可在一个载荷中得到视频帧和音轨；YuE2/Breeze 得到音轨。`batch_index=-1` 表示保留全部批次。解码器与采样器的 `model_type` 必须一致。 |
| **MLX VAE 解码（PIL）** `MlxVAEDecodeRawPIL` | `latents`，以及节点内的 `model_type`、`model_path`、`precision`、`quantize`、`batch_index`；H3 可选 `audio_vae` | 插件内部小写 `images` | 旧的一体式解码节点，为已有工作流保留。它把 VAE 选择 widget 内置在解码节点中；新工作流建议使用 `MlxVAELoader → MlxVAEDecoder`。 |
| **MLX PIL → 张量** `MlxPilToTorch` | 插件内部小写 `images` | 原生 `IMAGE`、`MASK`、`AUDIO` | 把载荷转换为 ComfyUI 可预览/保存的类型。图片/视频使用 `IMAGE`，YuE2、Breeze 和带声音的 H3 使用 `AUDIO`。纯音频载荷会给图片输出一个 1×1 黑图占位。 |
| **MLX 保存图片** `MlxSaveImage` | 插件内部小写 `images`、`filename_prefix`、`capture` | 无（输出节点） | 不经过 Torch，直接保存内部 PIL。实际目录固定为本仓库 `output/MlxSaveImage/<filename_prefix>/`，不是 ComfyUI 主目录的 `output/`。`capture=<auto>` 时写时间戳 PNG 及同名 JSON；自定义时写 `<capture>_<序号>.png` 且不写 JSON。只保存图片，不保存音频。 |

### 5.6 语音识别节点

| 节点 | 主要输入 | 输出 | 用途与限制 |
| --- | --- | --- | --- |
| **MLX Whisper 语音转文字** `MlxWhisperTranscribe` | 原生 `AUDIO`、本地 `model_path`、`language`、`temperature`、`condition_on_previous_text`、可选 `initial_prompt` | 纯文本 `text`、识别语言 `language` | 使用本地 MLX Whisper，固定做转写，不联网下载。模型候选仅扫描 `transformer/` 下配置有效的 Whisper checkpoint。Breeze 克隆时可把 `text` 直接接到 `MlxBreezeSampler.ref_text`；短参考音频通常用 `temperature=0`、关闭前文条件。 |

### 5.7 Loader 的延迟加载行为

三个 Loader 输出的是轻量配置 handle，不会立即把大模型读入统一内存。真正加载发生在：

- `MlxTextEncoder` / `MlxQwenEditEncoder`：加载文本编码器和 tokenizer；
- `MlxKSamplerMLX` / `MlxBreezeSampler`：加载主模型或 Transformer；
- `MlxVAEEncoder` / `MlxVAEDecoder`：加载 VAE。

因此“执行到 Loader 很快”是正常现象，首次执行后续消费节点才会出现明显加载时间。

### 5.8 不能混用原生 ComfyUI 模型对象

虽然 `CLIP` 等端口名与原生类型相同，MLX 节点实际传递的是自己的 handle：

- 原生 Checkpoint Loader 的 MODEL 不能接 MLX 采样器；
- 原生 CLIP 不能接 `MlxTextEncoder`；
- 原生 VAE 不能接 `MlxVAEEncoder` 或 `MlxVAEDecoder`。

请始终使用本插件的三个 MLX Loader。

## 6. 输出位置

### `MlxSaveImage`

`MlxSaveImage` 不使用 ComfyUI 的全局输出目录，而是写入本插件仓库：

```text
<ComfyUI-MLX-GEN>/output/MlxSaveImage/<filename_prefix>/
```

当 `capture=<auto>` 时，会同时写入同名 metadata JSON。

### ComfyUI 原生保存节点

- `MlxPilToTorch.images → SaveImage`：写入 ComfyUI 的正常图片输出目录；
- `MlxPilToTorch.audio → SaveAudio`：写入 ComfyUI 的正常音频输出目录；
- `MlxPilToTorch.images/audio → CreateVideo → SaveVideo`：写入 ComfyUI 的正常视频输出目录。

示例图片工作流常常同时连接 `MlxSaveImage` 和原生 `SaveImage`，所以同一结果可能保存两份。

## 7. LoRA 支持

LoRA 文件放到：

```text
<ComfyUI-MLX-GEN>/models/mlx/lora/
```

插件会递归扫描 `.safetensors` 文件，例如：

```text
lora/my_style.safetensors
lora/flux/my_character.safetensors
```

把 `MlxTransformerLoader.model` 接到一个或多个 `MlxModelLoraApply`，再把最后一个节点的
`model` 输出接到采样器。基础 Transformer 物化后，插件会按模型家族选择映射并应用权重；
应用 0 层、目标形状不符、文件缺失或只成功一部分都会报错，不会退回基础模型继续生成。

| 模型大类 | LoRA mapping / 格式 | 状态 |
| --- | --- | --- |
| `z_image` | mflux `ZImageLoRAMapping` | 支持普通浮点 LoRA |
| `flux2` | mflux `Flux2LoRAMapping` | 支持普通浮点 LoRA / LoKr（以 mapping 可识别键为准） |
| `qwen_image` / `qwen_edit` | mflux `QwenLoRAMapping` | 支持普通浮点 LoRA |
| `ideogram4` | mflux `Ideogram4LoRAMapping` | 支持；同时应用到条件/无条件 Transformer |
| `minimax_h3` | 插件内 H3 mapping | 支持 diffusers/LightX2V 拆分键、ComfyUI/ai-toolkit/kohya 原始融合键及 musubi 扁平键 |

MiniMax-H3 还支持 ComfyUI 的 `int8_tensorwise + convrot` LoRA：插件会按文件中的
`weight_scale` 和 `convrot_groupsize` 解码为 BF16，再拆融合 QKV、交换原始 SwiGLU 的
`[gate; value]` 行序并应用。DiffSynth-Studio 的 `.default.weight` 融合 QKV 是逐 head 交错
布局，目前会明确拒绝。对 H3 的 LightX2V Turbo LoRA，请按发布页建议使用 `strength=1.0`
和对应的 4/8 步采样设置；LoRA 只改变 Transformer，不会自动修改采样器步数。

当前限制：

- `MlxClipLoraApply` 尚不支持；文本编码器 LoRA 需要单独的 CLIP/Qwen/T5 映射；
- `yue2` 与 `breeze_tts2` 是音频生成 runtime，不接入 Transformer LoRA 节点；
- 图片家族暂不解码 Comfy `comfy_quant` LoRA；若检测到会报错，不能把 int8 整数当浮点矩阵；
- 原生 ComfyUI/Torch LoRA 节点仍不能连接本插件的小写 `model` MLX handle。

## 8. 常见问题

### 8.1 Loader 下拉框显示 `<无可用权重>`

依次检查：

1. 模型是否放在 `<ComfyUI-MLX-GEN>/models/mlx`，而不是普通 ComfyUI 模型目录；
2. 组件是否放在正确子目录；
3. 软链接目标是否存在：`ls -la /你的/ComfyUI/custom_nodes/ComfyUI-MLX-GEN/models/mlx/<组件>/`；
4. 目录名是否以 `.` 或 `__` 开头，这两类名称会被忽略；
5. 放好模型后是否重启了 ComfyUI。

### 8.2 报“模型大类不匹配”

检查 Transformer、CLIP、VAE 的 `model_type`。例如 Qwen 文生图必须全部是
`qwen_image`，Qwen 编辑必须全部是 `qwen_edit`。不能只修改其中一个 Loader。

### 8.3 报缺少 `config.json`

常见原因：

- 链接到了 Hugging Face cache 外壳，但该模型需要的是 snapshot 内的组件子目录；
- H3 的 Transformer 或 VAE 链接到了单个权重文件，旁边没有组件配置；
- YuE2 只复制了 `.safetensors`，没有保留同级配置和 tokenizer。

普通图片模型建议把 snapshot 下的 `transformer/`、`text_encoder/`、`tokenizer/`、`vae/`
分别软链接到对应组件目录。YuE2 和 Breeze 则要保留完整变体/checkpoint 目录。

### 8.4 Qwen 编辑报参考图尺寸与采样尺寸不一致

Qwen 编辑的参考 latent 与目标宽高强绑定。把 `MlxVAEEncoder` 的 `width`、`height` 输出连接到
`MlxKSamplerMLX` 的宽高输入，或者手工设置成完全相同的数值。

### 8.5 H3 只有画面没有声音

确认：

1. 使用的是 `minimax-h3-t2va.json`；
2. 第二个 `MlxVAELoader` 的 `role` 是 `audio_vae`；
3. 路径选的是音频 VAE 所在的 `MiniMax-H3`；
4. 它的输出接到了解码节点的 `audio_vae`；
5. `MlxPilToTorch.audio` 接到了 `CreateVideo.audio`。

### 8.6 内存不足或 Metal 报错

- 先使用 4-bit/8-bit 模型；
- `batch_size` 设为 1；
- 先降低分辨率；
- H3 先用 640×352、124 帧；
- YuE2 先用 `max_tokens=200`；
- 不要同时排队多个大模型工作流；
- 切换大型模型后，必要时重启 ComfyUI 以释放进程缓存。

如果报的是 Metal 层面的故障，按下面处理：

```text
[METAL] Command buffer execution failed: Caused GPU Timeout Error
        (00000002:kIOGPUCommandBufferCallbackErrorTimeout)
[METAL] Command buffer execution failed: Ignored (for causing prior/excessive GPU errors)
        (00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)
```

1. **第一条出现后必须重启 ComfyUI**，光重新排队没有用：Metal 上下文已经废了，之后每一次
   提交都会立刻返回上面那条 `Ignored (...SubmissionsIgnored)`（插件现在会把这两条翻译成
   中文提示并强调「先重启」，见 `h3/weights/loader.py` 的 `_metal_fault_hint`）。
2. 重启前把并行的重活关掉（ComfyUI 本体的 MPS 占用、其它大模型节点），内存被压进 swap 时
   GPU 命令更容易超时。
3. 插件的 H3 权重加载已按块 `mx.eval`（`EVAL_CHUNK`，默认 8 个张量一批）——整条 shard
   一次性求值会被 macOS 的 GPU 看门狗掐掉，这是 2026-09-20 修掉的老问题。
4. 反复超时就降档：`quantize` 用 q4、画布先用 640×352 甚至 256×256 冒烟。

### 8.7 为什么负向提示词没有效果

- FLUX.2 Klein 和 Z-Image Turbo 示例使用 guidance 1.0，通常不会启用额外 CFG 分支；
- Ideogram 4 固定使用自己的空无条件分支，忽略用户负向文本；
- H3 是 guidance 蒸馏模型，忽略通用负向条件；
- YuE2 的 negative 输入不是负向提示词，而是歌词；
- Breeze 推荐走专用采样器，不使用普通正负条件。

### 8.8 工作流导入后节点显示旧参数或连线错位

先确认 ComfyUI 加载的是当前仓库，而不是另一个同名插件副本。重启后重新导入
`workflows/` 中的最新 JSON；仍有问题时删除异常节点并从 `MLX/Gen` 菜单重新添加。

如果错位表现为**提交时报 `scheduler: 124` / `steps: 640` / `guidance: minimax_h3`
这类毫无道理的校验错误**，那是 seed 的伴随 widget 少了值：

- ComfyUI 前端对任何名叫 `seed` / `noise_seed` 的 widget 都会自动插一个
  `control_after_generate` 伴随 widget（与节点定义里有没有声明无关）；
- 所以工作流 JSON 的 `widgets_values` 必须在 seed 后面写上它的值
  （`fixed` / `increment` / `decrement` / `randomize`），漏写就会让后面**所有**
  widget 错位一格；
- 仓库里的工作流（含 `MlxKSamplerMLX` 的 22 份）已按这个规则序列化，
  `python tools/check_workflows_against_comfyui.py` 会挡住漏写的版本；
- 自己早先保存的工作流若报这个错，重新导入/保存一次（前端会自己补齐）即可。

### 8.9 同一个工作流突然慢了一倍以上（Z-Image 1024² 从 ~2s/步 变成 ~5s/步）

先查这一个环境变量，它是历史坑的元凶：

```bash
echo "$MLX_ENABLE_TF32"        # 期望：空（未设置 = MLX 默认开 TF32，快）

# 若要在 Python 里确认「插件导入是否动了环境」（在插件仓库根目录跑）
cd <插件目录>
PYTHONPATH=src python -c "import comfyui_mlx_gen, os; print(os.environ.get('MLX_ENABLE_TF32'))"   # 期望：None
```

注意：真正决定快慢的是 **ComfyUI 进程自己的环境**（shell 里的 `export`、
launchd/桌面启动器的环境、或 ComfyUI-Manager 里配的环境变量），
所以先确认 ComfyUI 是怎么启动的、有没有在它的环境里设过这个变量。

- M5 上 MLX **默认开 TF32**，而 TF32 关掉后 bf16/q8 的出图链路会慢 ~2.5 倍
  （Z-Image 1024²：2.0 → 5.1 s/步）。插件**不再**在导入时关它；只有 MiniMax-H3 的
  计算窗口内才会临时关（并打印 `[H3]` 取开关的日志）；
- 若你在 ComfyUI 启动环境里显式设了 `MLX_ENABLE_TF32=0`，出图就会一直慢——去掉它；
  想要 H3 音频最准而又不影响出图，请把 H3 放在独立进程 / 独立一次会话里跑；
- 若两条链路（本插件 vs `mlx-gen` CLI）**同窗口背靠背**跑都慢、且比例接近 1:1，
  那是机器状态（变频/温度）波动，不是插件问题。

完整复盘（含实测数据、5 分钟自检、复现命令）：
[`docs/TF32_SLOWDOWN_EXPLAINED.md`](docs/TF32_SLOWDOWN_EXPLAINED.md)。

### 8.10 二阶段放大没有更快 / 画面变软 / 音频变了

- **没更快**：收益来自「高分辨率只跑后半段 σ」。若 `stop_at_step` 设成 `0`（跑满）就
  没有任何收益（那等价于直接低分出片）；若 `start_at_sigma` 设得很大（如 0.92），
  高分辨率仍要跑很多步。默认 `stop_at_step=4 / steps=8` + `start_at_sigma=0.7 / steps=4`
  是折中档。
- **画面变软**：放大网络是 latent 空间里「学出来的」放大，不是凭空生成细节；放大倍率越大、
  `start_at_sigma` 越小，越像「平滑放大」。适当提高 `start_at_sigma`（0.8~0.9）会让二阶段
  重绘更多细节，同时把 `stop_at_step` / `steps` 一起加大让总分步数够。
- **首帧 / 尾帧变软**：接 `keyframes` 时锚点是从「已按低分画布拉伸过的图」再拉大的，
  比直接从原图编码略软；对首帧要求高时，要么让条件节点直接选目标画布（那首阶段也得同画布），
  要么二阶段不接 `keyframes`（纯 latent 精修，首帧可能轻微漂移）。
- **音频变了**：默认 `audio_mode=follow_video` 会跟视频一起收尾（音轨尾部会重绘）；想尽量
  沿用第一阶段的音轨选 `continue_stage1`（不重加噪），要彻底重画选 `regenerate`。
- **放大后画面混沌 / 像噪点被放大**：检查一阶段的 `output_mode` 是不是被改成了
  `latent (x_t)`。放大网络是在**干净 latent 对**上训练的，而 `stop_at_step=4` 时 x_t 里
  92% 是噪声（`x = σ·noise + (1-σ)·x0`，σ=0.9231）—— 放大网络会连噪声一起放大，二阶段
  再重采样就变成混沌。**保持 `denoised (x0)`** 即可（它复用最后一步的模型输出，不额外耗时）。
  自检方法：把 `MlxVAEDecoder` 直接接在 `MlxH3LatentUpscaler` 后面解码看一眼 —— 那一帧
  应该是「结构清晰、略软」。同时确认没有重复加 VAE mean/std；如果放大节点日志报告
  `输出幅度失控`，问题是 checkpoint/前向兼容性，节点会在进入 VAE 前拒绝该结果，
  不要用任意常数缩放去掩盖它。内置三线性插值只能手动选择做诊断，不能作为神经模型的
  自动 fallback。
- **内存吃紧 / 被系统 kill**：放大**不省内存** —— 二阶段仍在目标分辨率上跑，峰值与直出
  高分辨率接近；把 `scale` 降一档、或缩 `num_frames`。内存预检（>72% 警告、超限拒绝）
  在二阶段会用**新画布**重新算一次。

### 8.11 一阶段报 `shift must be positive, got 0.0`（或二阶段警告「起点高于一阶段停点」）

**`shift must be positive, got 0.0`** = `video_shift` / `audio_shift` 被设成了 0。H3 是**双整流流**
模型，两条 shift 必须 > 0：视频默认 **12.0**、音频默认 **3.0**（`0` 不是「不位移」，是非法值）。
把节点① 的 `video_shift` 改回 `12.0` 即可。新节点的滑杆最小值已经抬到 `0.1`，
并在进入模型加载前就给出中文报错；但**既有 `MlxKSamplerMLX` 的滑杆仍然允许 0**
（插件既有行为，本仓库不改它），那边保持默认值就行。

**二阶段起点高于一阶段停点**（例如一阶段 `stop_at_step=4`（σ=0.75）而二阶段
`start_at_sigma=0.9231`）：一阶段交给下游的是「该 σ 处的 x0 估计」，二阶段会按
`x = (1-σ)·x0 + σ·noise` 重新加噪 —— 起点比停点还高，等于把刚算出来的 x0 又推回更脏的状态，
放大 / 精修会明显变弱（日志里会打印 ⚠️ 提醒）。规则是：

> **两段步数之和 = 适配器标称步数；`start_at_sigma` 取「一阶段停下的那个 σ」或更低。**

各适配器的正确组合（`σ` 都是母网格上的点，二阶段子网格会与母网格后半段逐点一致）：

| 适配器 | 一阶段 | 二阶段 | 合计 | x0 在二阶段起点的权重 |
| --- | --- | --- | --- | --- |
| 4 步（v0.1 / v1.0） | `steps=4, stop_at_step=2`（σ 0.9231 / 0.75） | `steps=2, start_at_sigma=0.9231` | 4 | 7.7% |
| 8 步 | `steps=8, stop_at_step=4`（σ 0.9231 / 0.75） | `steps=4, start_at_sigma=0.9231` | 8 | 7.7% |
| 8 步省时档 | `steps=8, stop_at_step=6`（σ 0.8 / 0.5） | `steps=2, start_at_sigma=0.8` | 8 | **20%** |

> 反例（会把 x0 推回更脏的状态，日志会警告）：`steps=5, stop_at_step=4` + `steps=1,
> start_at_sigma=0.9231` —— 一阶段停在 σ=0.75，二阶段却从 0.9231 起。

**二次采样后是"雪花点"/彩色噪点**：先看一阶段 `output_mode` 是否为 `denoised (x0)`，
再看放大节点日志中的 normalized latent 统计。当前节点会检查神经输出的 finite、布局和
幅度；若报告 `输出幅度失控`，说明 checkpoint/前向不兼容，节点已在进入 VAE 前拒绝结果，
不要用任意常数缩放掩盖问题。`<内置：latent 三线性插值（不用模型）>` 只能手动选择做
诊断，不是神经模型的自动 fallback；两份示例工作流仍默认使用神经 3D checkpoint。
完整实测与复现步骤见 [`docs/MINIMAX_H3_VIDEO_IMPLEMENTATION.md`](docs/MINIMAX_H3_VIDEO_IMPLEMENTATION.md)
附录 F。

排除了放大网络之后，再按下面两条查（日志里都会有 ⚠️ 提示）：

1. **二阶段步数不够走完母网格的原生台阶**：例如 `start_at_sigma=0.9231` 只给 1 步 ——
   等于要求模型在 92% 噪声下**一步吐出成品**；蒸馏适配器在这个 σ 的预测本来就是"还有几步要走"的
   中间估计，直接解码就是雪花。8 步母网格在 0.9231 之后有 **4 段**原生台阶（至少要 4 步），
   4 步母网格有 **2 段**。
2. **基座模型步数太少**：H3 基座需要 **40~50 步**；`steps=5` 这种档位下交出的 x0 本身还是粗糙
   估计，放大后就是雪花。少步数必须配 4/8 步的 LightX2V Turbo 适配器（且把两段之和对齐到 4 / 8）。
   日志会打印：`⚠️ [H3 一阶段] 没挂加速 LoRA 却只跑 5 步：…基座官方档是 40~50 步…`。

> **实测确认（2026-09-20）**：二阶段从「`steps=1 @0.9231`（一步跳完）」改成「原生尾部」后，
> 雪花消失。能出片的 4 步档只有一种配法：**4 步加速 LoRA + 一阶段 `steps=4, stop_at_step=2`
> （σ 0.9231 / 0.75）+ 二阶段 `steps=2, start_at_sigma=0.9231`**（总 4 步，二阶段子网格
> `[0.9231, 0.8, 0]` 正是 4 步母网格的原生尾部）。**基座模型没有"4 步档"** —— 它需要 40~50 步；
> 用基座跑 4 步只能得到粗糙 x0 + 放大，不会雪花但也不会清晰。

## 9. 推荐检查清单

首次运行某个模型前，按以下顺序检查：

- [ ] ComfyUI 确实运行在 Apple Silicon 上；
- [ ] 依赖安装在 ComfyUI 实际使用的 Python 中；
- [ ] 模型根目录与 `paths.py` 的 `MODEL_ROOT` 一致；
- [ ] 每个组件软链接都没有断开；
- [ ] 放入模型后已经重启 ComfyUI；
- [ ] 同一工作流的 Loader 使用相同 `model_type`；
- [ ] Loader 选择的是同一权重集的对应组件；
- [ ] Qwen 文生图没有连接参考图；
- [ ] Qwen 编辑使用带图条件节点，并连接与采样器一致的宽高；
- [ ] H3 的 Transformer 和文本编码器选择 4/8-bit 量化；
- [ ] YuE2 的 negative 输入按“歌词”理解；
- [ ] Breeze 的目标文本使用外部 STRING 节点连接。
