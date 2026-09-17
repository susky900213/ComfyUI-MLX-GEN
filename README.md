# ComfyUI-MLX-GEN

面向 Apple Silicon 的 ComfyUI MLX 生成节点。模型组件通过纯数据 handle 在节点间传递，
Transformer、文本编码器和 VAE 只在实际消费它们的节点中延迟加载。

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

本项目的模型实现来自 `mflux`（发行包名见 `requirements.txt`），并依赖 Apple Silicon
上的 `mlx`。节点注册入口是仓库根目录的 `__init__.py`。

## 静态回归测试

Qwen-Image 专项测试不加载真实权重，覆盖模型登记、配置解析、Qwen Edit 构造参数过滤、
prompt/latent API、采样器边界，以及工作流节点、widget、socket 和 link 契约：

```bash
/opt/anaconda3/envs/py313/bin/python tests/test_qwen_image.py
```

测试成功时退出状态为 0；任何检查失败都会汇总失败项并以状态 1 退出。

## 其他示例工作流

`workflows/` 还包含 Z-Image、Flux.2 Klein、Qwen-Image-Edit 与 MiniMax-H3 示例。各工作流
序列化了对应模型的推荐采样参数；切换模型家族时请同时修改 Transformer、CLIP 和 VAE
加载器，避免混用不同权重集。