# ComfyUI-MLX-GEN

面向 Apple Silicon 的 ComfyUI MLX 生成节点。模型组件通过纯数据 handle 在节点间传递，
Transformer、文本编码器和 VAE 只在实际消费它们的节点中延迟加载。

完整的安装、模型目录、节点和示例工作流说明见：[中文使用手册](USAGE_ZH.md)。
Qwen-Image 2.1 用户可直接阅读：[Qwen-Image 2.1 下载、安装与使用指南](QWEN_IMAGE_21_USAGE_ZH.md)。

## Breeze-TTS-2 本地语音生成

Breeze-TTS-2 通过专用采样器在 **ComfyUI 进程内**完成文本编码、语音 token 生成与
audio tokenizer 解码，不启动服务、不调用 HTTP API。目标文本来自独立的 `STRING`
节点；`MlxBreezeSampler` 只暴露输入 socket，不在采样器内显示文本输入框：

```text
PrimitiveStringMultiline（要朗读的目标文本）
  → MlxTransformerLoader + MlxBreezeSampler
  → MlxVAELoader（兼容句柄）+ MlxVAEDecoder
  → MlxPilToTorch（AUDIO 输出，24 kHz 单声道）
  → SaveAudio / PreviewAudio
```

可直接导入默认使用内置 `S0` 说话人的示例：

```text
workflows/breeze-tts2.json
```

声音克隆示例已经预先连接目标台词、参考音频逐字稿和 `Load Audio`：

```text
workflows/breeze-tts2-voice-clone.json
```

导入后选择一段 3–10 秒的干净参考人声，并把“参考音频逐字稿”改成录音中实际说出的
完整内容；逐字稿必须尽量与录音一致。然后在“要生成的目标台词”中填写希望克隆音色
朗读的新内容，即可排队生成。

如果已经下载本地 MLX Whisper，可以导入自动转写版本；同一个 `Load Audio` 会同时送入
Whisper 和 Breeze，`MlxWhisperTranscribe.text` 会直接连接到 `ref_text`：

```text
workflows/breeze-tts2-voice-clone-asr.json

Load Audio ─┬→ MlxWhisperTranscribe → STRING → MlxBreezeSampler.ref_text
            └───────────────────────── AUDIO → MlxBreezeSampler.ref_audio
```

示例默认使用：

```text
<ComfyUI-MLX-GEN>/models/mlx/transformer/whisper-large-v3-mlx
```

`MlxWhisperTranscribe` 只扫描本地 `transformer/` 中同时包含 `config.json` 和
`weights.npz`/`weights.safetensors`、且 `config.json:model_type` 为 `whisper` 的目录；
不会因路径错误而静默联网下载。节点固定执行 `task=transcribe`，关闭 word timestamps，
首个输出只有可直接供 Breeze 使用的纯文本。普通话可把 `language` 固定为 `zh`；粤语用
`yue`；中英混合或未知语言可选 `auto`。`initial_prompt` 只用于提示人名、术语或产品名，
不要填写“请转写”等任务指令。3–10 秒参考音频建议保持
`temperature=0`、`condition_on_previous_text=false`。

节点会取第一批音频、将声道平均为单声道，并在内存中重采样到 Whisper 所需的 16 kHz，
不创建临时音频文件。转写完成后会主动释放 large-v3 及 MLX cache，再让 Breeze 加载，
避免两个大模型同时常驻。ASR 仍可能漏掉语气词、数字或专有名词；音色克隆质量优先时，
请检查识别结果，或改用上面的手工逐字稿工作流进行修正。

### 权重与依赖

下载 `mlx-community/Breeze-TTS-2-mlx-4bit`（也可使用 8-bit 或 BF16 变体），把**完整
checkpoint 目录**放入或软链接到 transformer 目录。目录必须保留根部的
`config.json`、模型权重及其文本/audio tokenizer 子目录；`config.json` 中的
`model_type` 应为 `breeze_tts`：

```text
<ComfyUI-MLX-GEN>/models/mlx/
└── transformer/Breeze-TTS-2-mlx-4bit -> <完整 Hugging Face snapshot>
```

Transformer 与 VAE 两个 Loader 的 `model_type` 都选 `breeze_tts2`，路径都选同一个
`Breeze-TTS-2-mlx-4bit`。Breeze checkpoint 已经包含主模型、文本编码器和 audio
tokenizer，因此 VAE Loader 仅传递兼容句柄，不会重复加载模型，也不需要再向 `vae/`
复制一份权重。真实量化精度由 checkpoint 决定；Loader 的 `quantize` 建议按所用变体
填写，插件不会进行二次量化。

Breeze runtime 固定为 `mlx-audio==0.5.1`，ASR runtime 固定为
`mlx-whisper==0.4.3`。必须在 **ComfyUI 实际使用的 Python
环境**中安装本仓库依赖并重启 ComfyUI：

```bash
cd /你的/ComfyUI/custom_nodes/ComfyUI-MLX-GEN
python -m pip install -r requirements.txt
```

### 三种生成模式

专用采样器只显示 Breeze 使用的生成参数。`text`、`ref_text` 与 `instruction` 都是
外部 `STRING` socket，不会在采样器节点内生成文本框；可连接 ComfyUI 核心
`PrimitiveStringMultiline` 或任意兼容的字符串输出。一次固定生成一条语音。

- **`speaker`（内置说话人）**：在 `speaker` 选择 `S0`–`S9`。官方模型只公开
  这十个标签，没有可靠的音色文字映射，因此本项目不臆测性别或年龄。
- **`voice_clone`（音色克隆）**：把 ComfyUI 原生 `Load Audio`（或任意 `AUDIO`
  输出）接到 `ref_audio`，并把另一个外部文本节点接到 `ref_text`，内容为参考音频的
  **逐字转写**。建议使用 3–10 秒干净人声。插件会取第一批、声道求均值为单声道，再
  用原始采样率写入临时 PCM16 WAV；`mlx-audio` 编码后该临时文件立即删除。
- **`voice_design`（音色设计）**：不要连接 `ref_audio`，把音色、情绪、语速或表达方式
  的外部文本接到 `instruction`。`cfg_scale != 1.0` 只在有 instruction 时
  启用模型的 CFG 分支。

`temperature`、`top_p`、`top_k`、`repetition_penalty` 和 `max_tokens` 会原样传给
mlx-audio 0.5.1；`max_tokens` 是语音帧上限（最大 750），模型可以提前结束。`seed`
控制采样随机性。

原有 `MlxKSamplerMLX` 节点仍保留，已有工作流和其他模型链路不受影响；新的 Breeze
示例与推荐链路使用 `MlxBreezeSampler`。

完整模型只在 waveform 缓存未命中时加载一次。生成结束后最终 24 kHz 波形进入独立缓存，
完整模型立即释放；VAE Decode 阶段只把该波形包装为 `AudioTrack`，不会再加载第二份模型。
克隆缓存键使用规范化波形内容及其原采样率的摘要，不包含随机临时文件名。

## YuE2-3B 本地音乐生成

YuE2 通过已有的通用 MLX 节点链在 **ComfyUI 进程内**完成文本规划、语义 codec
生成、NAR 声学 latent 合成和 VAE 解码；不启动服务，也不调用 HTTP API：

```text
MlxClipLoader
  → MlxTextEncoder（positive = style / 风格描述）
  → MlxTextEncoder（negative = lyrics / 歌词，不是负向提示词）
  → MlxTransformerLoader + MlxKSamplerMLX
  → MlxVAELoader + MlxVAEDecoder
  → MlxPilToTorch（AUDIO 输出）
  → SaveAudio / PreviewAudio
```

可直接导入示例工作流：

```text
workflows/yue2-3b.json
```

工作流使用 ComfyUI 内置的 `SaveAudio`（FLAC）与 `PreviewAudio`，默认将文件写到
ComfyUI 输出目录的 `audio/` 子目录。当前上游已把旧 `SaveAudio` 标为 deprecated，
但仍保留兼容；如果所用 ComfyUI 版本提供新的音频保存节点，也可以直接把
`MlxPilToTorch.audio` 接过去。

### YuE2 权重放置

从 `npario/YuE2-3B-MLX` 下载转换后的 checkpoint。每个 `4bit/`、`8bit/` 或
`bf16/` 精度变体目录都应完整保留以下文件：

```text
config.json
model.safetensors
qwen.tiktoken
yue2_generation_config.json
vae_config.json
vae.safetensors
```

推荐把同一个完整变体目录分别软链接到 `transformer/` 与 `vae/`。示例工作流默认使用
`YuE2-3B-MLX-4bit`：

```text
<ComfyUI-MLX-GEN>/models/mlx/
├── transformer/YuE2-3B-MLX-4bit -> <HF snapshot>/4bit
└── vae/YuE2-3B-MLX-4bit         -> <HF snapshot>/4bit
```

也兼容单文件软链接，但链接必须直接指向变体目录内的 `model.safetensors` 或
`vae.safetensors`，以便插件从目标文件的同级目录找回配置、tokenizer 和另一组权重。
例如：

```text
transformer/YuE2-3B-MLX-4bit.safetensors -> <HF snapshot>/4bit/model.safetensors
vae/YuE2-3B-MLX-4bit-vae.safetensors     -> <HF snapshot>/4bit/vae.safetensors
```

使用单文件链接时，导入工作流后要在三个 Loader 中改选对应名称。三个 Loader 的
`model_type` 必须全部为 `yue2`。YuE2 的真实精度由所选 checkpoint 变体决定；Loader
里的 `precision` / `quantize` 值只参与句柄诊断和缓存键，不会把 checkpoint 再量化一次。

### 提示词与采样参数

- positive 文本是 **style**：填写曲风、乐器、速度、情绪、演唱风格等，例如
  `cinematic synthwave, warm female vocal, 100 BPM, wide stereo`；不能为空。
- negative 文本是 **lyrics**，并非需要排除的内容。可以使用 `[Verse]`、`[Chorus]`
  等段落标记；留空表示纯音乐。
- `cot=off` 跳过 ABC 规划并直接生成 codec，启动更快；`melody` 先规划不带和弦的
  旋律 ABC；`full` 先规划带和弦的完整 ABC，规划时间也最长。
- `max_tokens` 是语义 codec token 上限，每个 token 约对应 40 ms，所以 200 约为
  8 秒、1500 约为 1 分钟、9000 理论上约为 6 分钟。模型可能提前生成结束标记，
  因而这是上限而不是保证时长；建议先用 200 验证工作流。
- `steps` 控制 NAR midpoint ODE 的步数（默认 32），影响声学 latent 的计算量与质量，
  不控制时长。`scheduler` 固定为 `yue2_midpoint`。
- `guidance=1.0` 不额外运行 CFG 分支；允许范围为 1.0–5.0。`batch_size` 必须为 1。
  `width`、`height`、视频帧数/shift 和 `kv_cache` 是通用采样器为其他模型保留的 widget，
  YuE2 不使用这些值。

### 精度与内存

- 首次使用优先选择 `4bit`；`8bit` 和 `bf16` 需要更多统一内存。
- 长音频会增加自回归 KV cache、语义 token 和声学 latent 的内存与耗时；先从
  `max_tokens=200` 开始，再逐步增加。
- 采样结束后插件会主动释放 YuE2-3B 主模型，再加载 VAE；解码结束后也会释放 VAE，
  避免二者同时常驻。声学 latent 保留在小型缓存中，因此参数完全相同的重复执行可直接
  命中 latent，不会为了重建句柄再次加载 3B 主模型。修改 seed、style、lyrics、`cot`、
  `max_tokens`、steps 或 guidance 都会产生新的缓存键。
- 已验证 4-bit 端到端链路可将 `[200, 64]` latent 解码为 48 kHz 立体声；实际听感仍应
  结合目标提示词和音频设备人工试听。

## MiniMax-H3 与 PipeNetwork 预量化 Transformer

仓库提供十四份可直接导入的 MiniMax-H3 工作流（后七份由 `tools/gen_h3_workflows.py`
生成，最后两份由 `tools/gen_h3_two_stage_workflow.py` 生成，改完脚本重跑即可覆盖）：

```text
workflows/minimax-h3-t2va.json                    # 文生视频 + 立体声（Base）
workflows/minimax-h3-t2v-no-audio.json            # 文生视频，仅画面（Base）
workflows/minimax-h3-i2va-first-frame.json        # 首帧生视频（Base）
workflows/minimax-h3-i2va-last-frame.json         # 尾帧生视频（Base）
workflows/minimax-h3-i2va-first-last-frame.json   # 首尾帧生视频（Base）
workflows/minimax-h3-single-image-to-video.json   # 1 张图钉成首帧（Base）
workflows/minimax-h3-multi-image-to-video.json    # 3 张参考图，第 1 张钉成首帧（H3-REF）
workflows/minimax-h3-reference-only-to-video.json # 3 张纯参考，不钉锚点（H3-REF）
workflows/minimax-h3-all-reference-to-video.json  # 4 张纯参考 + 4 步加速 LoRA（H3-REF）
workflows/minimax-h3-video-continuation-keep-audio.json     # 源视频 + 原声拼成片（Base）
workflows/minimax-h3-video-continuation-drop-audio.json     # 只出新片段（Base）
workflows/minimax-h3-video-continuation-replace-audio.json  # 新片段 + 外部配乐（Base）
workflows/minimax-h3-two-stage-upscale.json       # 二阶段：低分采样 → latent 放大 → 高分精修
workflows/minimax-h3-two-stage-upscale-lora.json  # 同上 + 8 步加速 LoRA（总步数 4+4=8）
```

最后两份是**二阶段放大**链路，用到三个专用节点（既有节点一行都不用改）：
`MlxH3FirstPassSampler`（低分辨率跑到母网格第 k 步）→ `MlxH3LatentUpscaler`
（只给倍率，目标画布由上一段推导，3D 网络在 latent 空间放大）→
`MlxH3SecondPassSampler`（在目标分辨率上跑完剩余 σ，含音频三模式）。
实测 `640×352 → 1280×704` 的 2× 放大约 4 秒，整体比直出高分辨率省约 35–40% 时间。

带 `-lora` 的那份在**一阶段与二阶段共用的那条 model 线**上插了一个 `MlxModelLoraApply`
（8 步 FL2VA 适配器，`strength=1.0`），并把两段配成 `4 + 4 = 8`、切点取母网格第 4 点
（σ=0.9231）—— 这样两段合起来仍是适配器训练时的完整 8 步轨迹，只是后半段在 2× 分辨率上跑。
**换 4 步适配器时务必把两段一起改成 `2 + 2`**（详见
[`USAGE_ZH.md`](USAGE_ZH.md) §4.5 与
[`docs/MINIMAX_H3_VIDEO_IMPLEMENTATION.md`](docs/MINIMAX_H3_VIDEO_IMPLEMENTATION.md) 附录 E）。

默认工作流使用 640×352、124 帧（24 fps）和 50 步；只有
`minimax-h3-all-reference-to-video.json` 例外——它把「MLX 模型 LoRA」接在
transformer 与采样器之间，用 `minimax_h3_ref2v_lightx2v_turbo_4step_v0.1_resized_avg_rank_20_bf16.safetensors`
并把 steps 改成 4（导进去前确认 `models/mlx/lora/` 里确实有这个文件；
同目录的 `..._fl2v_...` 是首 / 尾帧（FL2VA）任务用的，`..._taomate_3step_...`
只写了步数、没写适用任务，都别照抄参考工作流换过来）。H3 的 transformer、Qwen3-VL
文本编码器、tokenizer、视频 VAE 与音频 VAE 是五个独立组件；常规目录布局如下：

```text
<ComfyUI-MLX-GEN>/models/mlx/
├── transformer/MiniMax-H3/          # 或 MiniMax-H3-MLX-8bit（预量化，见下）
├── transformer/MiniMax-H3-ref/      # 参考生视频（Ref2VA）用的 REF transformer
├── text_encoder/MiniMax-H3/         # text_encoder_back
├── tokenizer/MiniMax-H3/
├── vae/MiniMax-H3/                  # 视频 VAE
├── audio_vae/MiniMax-H3/            # 音频 VAE
└── lora/                            # H3 加速适配器（ref2v 4 步、fl2v 4/8 步、taomate 3 步）
```

**参考生视频必须选 Minimax-H3-REF。** 官方 checkpoint 的 transformer 有两条：
`transformer`（Base）只支持 0~2 张图、而且只认首 / 尾两个 latent 锚点槽；
`transformer_ref`（H3-Base-Ref2VA，落到本地即 `transformer/MiniMax-H3-ref`）
才支持最多 9 张参考图与「按参考生成」。因此「多图参考」「纯参考」这类
工作流的 `MlxTransformerLoader` 都要选 `MiniMax-H3-ref`（或
`MiniMax-H3-Ref2VA-MLX-Serve-8bit.safetensors`），而首 / 尾帧、视频续写这类
锚点任务仍用 Base 的 `MiniMax-H3`。文本编码器、tokenizer 与两个 VAE 两档共用
同一份 `MiniMax-H3`，不需要另配。条件与权重不配对时（比如拿 Base 跑 3 张
参考图，或不钉锚点的纯参考）`MlxKSamplerMLX` 会直接报错，不会静默出垃圾。

Transformer 也可以直接使用
[`pipenetwork/MiniMax-H3-MLX-8bit`](https://huggingface.co/pipenetwork/MiniMax-H3-MLX-8bit)
的原生 MLX 预量化 checkpoint，而不需要先反量化再重新量化。它只替换上面第一项：

```text
<ComfyUI-MLX-GEN>/models/mlx/transformer/
└── MiniMax-H3-MLX-8bit -> <HF cache>/models--pipenetwork--MiniMax-H3-MLX-8bit
```

软链接可以指向完整 Hugging Face cache 外壳，不必手工定位 commit 目录。插件会读取
`refs/main` 并解析到对应的 `snapshots/<revision>/`；如果引用缺失、revision 非法、snapshot
不存在或为空，会立即报告 cache 损坏。直接链接到 snapshot 或普通模型目录仍保持原行为。

加载 PipeNetwork checkpoint 时，插件会在读取权重前严格校验 `quant_config.json` 与
safetensors header，然后按仓库声明的 `bits`、`group_size` 和 AdaLN 位宽重建 MLX
`QuantizedLinear` 模块。磁盘中的 packed `weight`、`scales`、`biases` 会逐 shard 直接写入，
不会反量化，也不会二次量化；fused QKV 和 MLP 只做布局与模块名转换。此时 Transformer
Loader 的 `quantize` 选择不会改变 checkpoint 精度，建议仍选择 `8`，使工作流配置与实际
q8 权重一致。文本编码器与两个 VAE 仍使用各自原有的加载/量化设置。

导入工作流后，把 `MlxTransformerLoader` 的路径改成 `MiniMax-H3-MLX-8bit`；CLIP、
tokenizer、视频 VAE 与音频 VAE 继续选择 `MiniMax-H3`。完整 8-bit transformer 约 35 GB，
按 7 个 shard 增量加载；最终内存还要叠加文本编码器、VAE、MLX cache 和采样激活，请先用
示例的 640×352 / 124 帧做冒烟验证。

## Ideogram 4 FP8 本地文生图

Ideogram 通过现有 ComfyUI MLX 节点链运行，**不调用 API**，也没有单独的 Ideogram
生成节点：

```text
MlxClipLoader
  → MlxTextEncoder（正向 / 负向）
  → MlxKSamplerMLX
  → MlxVAELoader + MlxVAEDecoder
  → MlxPilToTorch
```

可直接导入示例工作流：

```text
workflows/ideogram-4-fp8.json
```

三个加载器的 `model_type` 都选择 `ideogram4`，权重目录选择
`ideogram-4-fp8`。官方 checkpoint 有五个本地组件：

```text
<ComfyUI-MLX-GEN>/models/mlx/
├── transformer/ideogram-4-fp8/               # conditional transformer
├── unconditional_transformer/ideogram-4-fp8/ # unconditional transformer
├── text_encoder/ideogram-4-fp8/
├── tokenizer/ideogram-4-fp8/
└── vae/ideogram-4-fp8/
```

每项都可以是指向 Hugging Face snapshot 对应子目录的软链接。如果
`transformer/ideogram-4-fp8` 指向完整 snapshot 下的 `transformer/`，插件也会自动查找
同级的 `unconditional_transformer/`，不强制再建第二个软链接。

推荐设置：

- Transformer / CLIP / VAE 的 `quantize` 均为 `0`，保留 checkpoint 原生 FP8/BF16；
- 1024×1024，batch 1；
- Scheduler 选 `ideogram4_default`（官方 `V4_DEFAULT_20`）；另有
  `ideogram4_quality`（48 步）和 `ideogram4_turbo`（12 步）；
- Ideogram 预设自带步数和逐步 guidance schedule，因此采样器的 `steps` / `guidance`
  对本模型仅作显示，执行时以所选预设为准；
- 正向提示词优先使用官方结构化 JSON caption。普通文本也能运行，但通常效果较弱；
- 负向节点仍需连接以保持标准 KSampler 工作流结构，但 Ideogram 4 固定使用自己的空
  无条件分支，节点中的负向文本会被忽略。

当前本地 MFLUX 实现只支持 Ideogram 4 **文生图**，不支持 Remix、参考图或蒙版编辑；
也不包含云端 Magic Prompt。模型是 gated 权重，需先在 Hugging Face 接受许可并自行下载。

## Qwen-Image 2.1 原生文生图与图片编辑

Qwen-Image 2.1 使用独立的 `qwen_image_21` 大类，不会回退到旧版 2512/2511 架构。
模型下载、约 31 GB 完整性检查、四组件软链接命令和图文操作步骤见
**[Qwen-Image 2.1 下载、安装与使用指南](QWEN_IMAGE_21_USAGE_ZH.md)**。

官方模型地址：<https://huggingface.co/Qwen/Qwen-Image-2.1>。下载后需要建立以下组件布局；
官方仓库的 tokenizer 位于 `processor/`，因此本地 `tokenizer/Qwen-Image-2.1` 应指向它：

```text
<ComfyUI-MLX-GEN>/models/mlx/
├── transformer/Qwen-Image-2.1 -> <官方模型>/transformer
├── text_encoder/Qwen-Image-2.1 -> <官方模型>/text_encoder
├── tokenizer/Qwen-Image-2.1 -> <官方模型>/processor
└── vae/Qwen-Image-2.1 -> <官方模型>/vae
```

仓库提供两份可直接导入的工作流：

```text
workflows/qwen-image-2.1.json       # 纯文生图
workflows/qwen-image-2.1-edit.json  # 原生参考图编辑
```

三个加载器统一选择 `qwen_image_21` 和 `Qwen-Image-2.1`。默认使用 40 步、
`flow_match_euler_discrete`、guidance 1.0，并支持 RGBA 解码。

编辑路径不是 legacy `qwen_edit` latent 的兼容层，而是官方 2.1 统一契约：同一批预处理
像素同时送入 Qwen3-VL 视觉塔和 2.1 VAE encoder；DiT 将参考 latent 插入视觉槽，使用
block-causal attention 和 prefix KV cache。单图工作流的关键连线为：

```text
LoadImage ───────────────→ MlxVAEEncoder.images
MlxVAELoader ────────────→ MlxVAEEncoder.vae
MlxVAEEncoder.ref_images ├→ 正向 MlxTextEncoder.ref_images
                          ├→ 负向 MlxTextEncoder.ref_images
                          └→ MlxKSamplerMLX.ref_images
```

正向、负向和采样器必须连接**同一个** `ref_images`，插件会在加载 DiT 前校验缓存键。
代码最多支持 10 张参考图；异尺寸多图可用 `MlxRefImageSet → MlxVAEEncoder.ref_source`。
`auto` 会保持每张图宽高比并缩放到约 `width × height` 的面积，再对齐到 32 像素倍数。

## Qwen-Image 2512 文生图

仓库提供可直接导入 ComfyUI 的示例工作流：

```text
workflows/qwen-image-2512.json
```

工作流默认参数：

- 模型大类：`qwen_image`
- 权重目录：`qwen-image-2512-8bit`
- 分辨率：1024×1024
- 采样步数：20
- Guidance：4.0
- Scheduler：`flow_match_euler_discrete`
- Transformer compile：关闭（Qwen Transformer 的调用参数不支持当前编译路径）

四类组件应放在插件使用的模型根目录下，并使用相同的权重集目录名：

```text
<ComfyUI-MLX-GEN>/models/mlx/
├── transformer/qwen-image-2512-8bit/
├── vae/qwen-image-2512-8bit/
├── text_encoder/qwen-image-2512-8bit/
└── tokenizer/qwen-image-2512-8bit/
```

也可以使用指向实际权重目录的绝对软链接。

在工作流的三个加载器中分别选择：

| 节点 | `model_type` | 权重路径 |
| --- | --- | --- |
| MLX CLIP 加载 | `qwen_image` | `qwen-image-2512-8bit` |
| MLX 模型加载 | `qwen_image` | `qwen-image-2512-8bit` |
| MLX VAE 加载 | `qwen_image` | `qwen-image-2512-8bit` |

`qwen_image` 是纯文生图链路：文本条件使用两个 `MlxTextEncoder` 节点，分别连接采样器
的 positive 和 negative；不要连接 `ref_images`。`qwen-image-2512-8bit` 不挂视觉塔，
采样器会明确拒绝参考图输入。

参考图编辑继续使用原有链路：三个加载器选择 `qwen_edit` +
`qwen-image-edit-2511-8bit`，条件使用 `MlxQwenEditEncoder`，并且必须连接
`MlxVAEEncoder` 输出的 `ref_images`。文生图支持不会改变这条编辑路径的行为。

## 依赖

使用 Python 3.13 环境安装：

```bash
python -m pip install -r requirements.txt
```

图像模型实现来自 `mflux==0.19.1`，Breeze-TTS-2 runtime 来自
`mlx-audio==0.5.1`（发行包名见 `requirements.txt`），并依赖 Apple Silicon 上的
`mlx>=0.32.0,<0.33.0`。节点注册入口是仓库根目录的 `__init__.py`。

**不要同时安装 `mflux` 与 `mlx-gen` 两个发行包。** 它们都会安装同名的 `mflux`
Python 模块，但版本约束不同：本仓库固定的 `mflux==0.19.1` 需要 MLX 0.32.x，而
`mlx-gen==0.36.0` 要求 `mlx<0.32.0`。从旧环境迁移时建议先移除两个
发行包，再按本仓库依赖重新安装：

```bash
python -m pip uninstall -y mlx-gen mflux
python -m pip install -r requirements.txt
python -m pip show mlx mflux
```

最后一条应显示 `mlx 0.32.x` 与 `mflux 0.19.1`，且 `python -m pip show mlx-gen`
应显示未安装。请务必在 **ComfyUI 实际使用的 Python 环境**中执行这些命令。

## Transformer LoRA

`MlxModelLoraApply` 已接入真实推理链路。LoRA 文件放在
`<ComfyUI-MLX-GEN>/models/mlx/lora/`，可串联多个节点后再接
`MlxKSamplerMLX.model`。支持以下模型家族：

- Z-Image：`ZImageLoRAMapping`；
- FLUX.2 Klein：`Flux2LoRAMapping`；
- Qwen Image / Qwen Edit：`QwenLoRAMapping`；
- Ideogram 4：`Ideogram4LoRAMapping`，同时应用到条件和无条件 Transformer；
- MiniMax-H3：插件内专用 mapping，包含 LightX2V/diffusers、原始融合 QKV/MLP、
  musubi-tuner 键和 ComfyUI `int8-convrot` LoRA 解码。

LoRA 在基础权重加载和量化完成后应用，并保留为量化 Linear 外层的低秩分支；不同路径、
顺序与强度参与缓存键。`strength=0` 完全跳过。CLIP/text-encoder LoRA、YuE2 和
Breeze-TTS-2 LoRA 当前不支持，并会明确报错而不是静默生成基础模型结果。详细格式和限制见
[`USAGE_ZH.md` 第 7 节](USAGE_ZH.md#7-lora-支持)。

## 静态回归测试

MiniMax-H3、YuE2、Breeze-TTS-2、Ideogram 4 与 Qwen-Image 专项测试都不加载真实大权重，
覆盖模型登记、组件路径、prompt/latent API、采样器边界，以及工作流节点、widget、socket
和 link 契约：

```bash
/opt/anaconda3/envs/py313/bin/python tests/test_qwen_image.py
/opt/anaconda3/envs/py313/bin/python tests/test_ideogram.py
/opt/anaconda3/envs/py313/bin/python tests/test_h3_pipenetwork.py
/opt/anaconda3/envs/py313/bin/python tests/test_h3_keyframes.py
/opt/anaconda3/envs/py313/bin/python tests/test_yue2.py
/opt/anaconda3/envs/py313/bin/python tests/test_breeze.py
/opt/anaconda3/envs/py313/bin/python tests/test_lora.py
/opt/anaconda3/envs/py313/bin/python tools/check_workflows_against_comfyui.py
```

`tools/check_workflows_against_comfyui.py` 会按 ComfyUI 与本插件的真实节点定义核对
每份工作流的槽名、类型与 widget 取值（ComfyUI 自带节点只查存在性），并检查
「视觉条件 → 该用哪一档 transformer」是否与声明一致；其中 `widgets_values` 会按
**前端真正渲染出来的 widget 列表**比对 —— `seed` 后面那个前端自动插的
`control_after_generate` 也必须写进 JSON，漏写就会让整排 widget 错位一格
（详见 `USAGE_ZH.md` §8.8）；`tools/gen_h3_workflows.py`
则用于重新生成后七份 H3 工作流。

测试成功时退出状态为 0；任何检查失败都会汇总失败项并以状态 1 退出。

## 其他示例工作流

`workflows/` 还包含 Z-Image、Flux.2 Klein、Qwen-Image-Edit、Ideogram 4、MiniMax-H3
、YuE2 与 Breeze-TTS-2 示例。各工作流序列化了对应模型的推荐采样参数；切换模型家族时请同时修改
Transformer、CLIP 和 VAE 加载器，避免混用不同权重集。
