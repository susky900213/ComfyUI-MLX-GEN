# ComfyUI-MLX-GEN

面向 Apple Silicon 的 ComfyUI MLX 生成节点。模型组件通过纯数据 handle 在节点间传递，
Transformer、文本编码器和 VAE 只在实际消费它们的节点中延迟加载。

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
/Users/apple/ComfyUI-Shared/models/mlx/
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
/Users/apple/ComfyUI-Shared/models/mlx/
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

本项目的模型实现来自 `mflux==0.19.1`（发行包名见 `requirements.txt`），并依赖 Apple
Silicon 上的 `mlx>=0.32.0,<0.33.0`。节点注册入口是仓库根目录的 `__init__.py`。

## 静态回归测试

Ideogram 4 与 Qwen-Image 专项测试都不加载真实大权重，覆盖模型登记、组件路径、官方预设、
prompt/latent API、采样器边界，以及工作流节点、widget、socket 和 link 契约：

```bash
/opt/anaconda3/envs/py313/bin/python tests/test_qwen_image.py
/opt/anaconda3/envs/py313/bin/python tests/test_ideogram.py
```

测试成功时退出状态为 0；任何检查失败都会汇总失败项并以状态 1 退出。

## 其他示例工作流

`workflows/` 还包含 Z-Image、Flux.2 Klein、Qwen-Image-Edit、Ideogram 4 与 MiniMax-H3 示例。各工作流
序列化了对应模型的推荐采样参数；切换模型家族时请同时修改 Transformer、CLIP 和 VAE
加载器，避免混用不同权重集。
