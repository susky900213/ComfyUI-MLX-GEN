"""YuE2 ABC 文件读取节点。"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ..yue2_melody import validate_abc


DEFAULT_ABC_FILE = "abc/your_song.abc"


def _input_directory() -> Path:
    """返回 ComfyUI input 目录；脱离 ComfyUI 时给出可测试的本地 fallback。"""
    try:
        import folder_paths
    except ModuleNotFoundError:
        return Path.cwd() / "input"
    return Path(folder_paths.get_input_directory())


def resolve_abc_file(filename: str) -> Path:
    """把 input/ 下的相对文件名解析为安全路径。"""
    filename = str(filename).strip()
    if not filename:
        raise ValueError("ABC 文件名不能为空；请填写 ComfyUI input/ 下的相对路径")
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("ABC 文件必须位于 ComfyUI input/ 目录内，不能使用绝对路径或 ..")
    if relative.suffix.casefold() != ".abc":
        raise ValueError(f"ABC 文件必须使用 .abc 扩展名：{filename!r}")

    root = _input_directory().expanduser().resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("ABC 文件必须位于 ComfyUI input/ 目录内") from exc
    if not path.is_file():
        raise FileNotFoundError(f"未找到 ABC 文件：{path}")
    return path


class MlxYue2LoadABC:
    """从 ComfyUI ``input/`` 目录读取并校验 YuE2 ABC 文件。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "abc_file": (
                    "STRING",
                    {
                        "default": DEFAULT_ABC_FILE,
                        "tooltip": "ComfyUI input/ 下的 .abc 相对路径，例如 abc/your_song.abc",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("abc",)
    FUNCTION = "load"
    CATEGORY = "MLX/Audio"

    @classmethod
    def VALIDATE_INPUTS(cls, abc_file, **kwargs):
        try:
            path = resolve_abc_file(abc_file)
            validate_abc(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, ValueError) as exc:
            return str(exc)
        return True

    @classmethod
    def IS_CHANGED(cls, abc_file, **kwargs):
        path = resolve_abc_file(abc_file)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def load(self, abc_file):
        path = resolve_abc_file(abc_file)
        text = path.read_text(encoding="utf-8-sig")
        validate_abc(text)
        # 统一换行，避免不同平台的 CRLF 改变 YuE2 prompt cache key。
        return (re.sub(r"\r\n?", "\n", text),)