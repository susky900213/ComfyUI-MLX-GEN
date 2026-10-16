"""Match downloaded images to scrambled HTML album pages and restore them.

The album pages handled by this node keep the source image in an ``img`` tag,
but display a visible canvas assembled from horizontal image slices.  The
canvas ``data`` attribute contains a base64 encoded JSON object with the
source filename and the equivalent ``drawImage`` arguments.  This module
matches already-downloaded local images to that filename and performs the
same composition with Pillow.

Only the Python standard library and Pillow are used so the node does not add
another HTTP or HTML parsing dependency to the custom-node package.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import shutil
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from PIL import Image, UnidentifiedImageError

from ..types import MlxPilImage


TARGET_CLASSES = frozenset({"center", "scramble-page", "spnotice_chk"})
IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
_PAGE_NUMBER_RE = re.compile(r"(?:^|[^0-9])([0-9]{1,})")
_DOWNLOADED_NAME_PREFIX_RE = re.compile(r"^imgi_[0-9]+_", re.IGNORECASE)
_PICKER_INPUT_PREFIX = "__comfy_input__/"
_PICKER_UPLOAD_ROOT = "mlx_html_scramble"


@dataclass
class _ScrambleBlock:
    """The relevant attributes collected from one target ``div``."""

    order: int
    element_id: str = ""
    page: int | None = None
    image_attrs: dict[str, str] = field(default_factory=dict)
    canvas_data: str = ""


class _AlbumHTMLParser(HTMLParser):
    """Collect target divs without depending on BeautifulSoup/lxml."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[_ScrambleBlock] = []
        self._div_stack: list[_ScrambleBlock | None] = []
        self.canonical_url = ""
        self.base_href = ""

    @staticmethod
    def _attrs_dict(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
        return {key.casefold(): value or "" for key, value in attrs}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_name = tag.casefold()
        values = self._attrs_dict(attrs)

        if tag_name == "div":
            classes = frozenset(values.get("class", "").split())
            block: _ScrambleBlock | None = None
            if TARGET_CLASSES.issubset(classes):
                block = _ScrambleBlock(
                    order=len(self.blocks),
                    element_id=values.get("id", ""),
                )
                self.blocks.append(block)
            self._div_stack.append(block)
            return

        if tag_name == "base" and not self.base_href:
            self.base_href = values.get("href", "").strip()
        elif tag_name == "link":
            rel = frozenset(values.get("rel", "").casefold().split())
            if "canonical" in rel and not self.canonical_url:
                self.canonical_url = values.get("href", "").strip()
        elif tag_name == "meta" and not self.canonical_url:
            property_name = values.get("property", "").casefold()
            name = values.get("name", "").casefold()
            if property_name == "og:url" or name == "og:url":
                self.canonical_url = values.get("content", "").strip()

        if not self._div_stack:
            return
        block = next((item for item in reversed(self._div_stack) if item is not None), None)
        if block is None:
            return
        if tag_name == "img" and not block.image_attrs:
            block.image_attrs = values
        elif tag_name == "canvas" and not block.canvas_data:
            block.canvas_data = values.get("data", "").strip()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "div" and self._div_stack:
            self._div_stack.pop()


def _page_number(block: _ScrambleBlock) -> int | None:
    """Extract a numeric page number from ``data-page``, id, or image id."""

    data_page = block.image_attrs.get("data-page", "").strip()
    if data_page.isdigit():
        return int(data_page)
    for value in (block.element_id, block.image_attrs.get("id", "")):
        match = _PAGE_NUMBER_RE.search(value)
        if match:
            return int(match.group(1))
    return None


def _ordered_blocks(blocks: list[_ScrambleBlock]) -> list[_ScrambleBlock]:
    """Prefer the page number but keep source order for ties/missing numbers."""

    for block in blocks:
        block.page = _page_number(block)
    return sorted(
        blocks,
        key=lambda block: (
            0 if block.page is not None else 1,
            block.page if block.page is not None else block.order,
            block.order,
        ),
    )


def _decode_canvas_data(value: str) -> dict[str, Any]:
    """Decode the base64 JSON stored in a canvas ``data`` attribute."""

    encoded = value.strip()
    if not encoded:
        raise ValueError("canvas data 为空")

    candidates = [encoded]
    padding = "=" * (-len(encoded) % 4)
    candidates.append(encoded + padding)
    decoded_text = ""
    for candidate in candidates:
        try:
            raw = base64.b64decode(candidate, altchars=b"-_", validate=False)
            decoded_text = raw.decode("utf-8")
            break
        except (binascii.Error, UnicodeDecodeError):
            continue
    if not decoded_text:
        # Supporting plain JSON makes the parser useful with hand-generated
        # HTML and costs nothing for the normal base64 form.
        decoded_text = encoded

    try:
        spec = json.loads(decoded_text)
    except (TypeError, ValueError) as exc:
        raise ValueError("canvas data 不是有效的 Base64 JSON") from exc
    if not isinstance(spec, dict):
        raise ValueError("canvas data JSON 顶层必须是对象")
    return spec


def _positive_int(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 不是整数：{value!r}") from exc
    if result <= 0:
        raise ValueError(f"{label} 必须大于 0：{value!r}")
    return result


def _image_paths(directory: Path) -> list[Path]:
    """Return supported downloaded images from the selected directory."""

    root = directory.expanduser()
    if not root.is_dir():
        raise ValueError(f"图片目录不存在或不是目录：{root}")
    paths = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )
    if not paths:
        extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise ValueError(f"图片目录没有可读取的图片：{root}（支持：{extensions}）")
    return paths


def _reference_filename(value: str) -> str:
    """Extract a filename from an HTML/Canvas URL or local path."""

    raw = unquote(str(value or "").strip())
    if not raw:
        return ""
    parsed = urlparse(raw)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raw = unquote(parsed.path)
    return Path(raw).name


def _comfy_directory(name: str) -> Path:
    """Return a ComfyUI input/output directory, with a test fallback."""

    try:
        import folder_paths
    except ModuleNotFoundError:
        return Path.cwd() / name
    getter = getattr(folder_paths, f"get_{name}_directory")
    return Path(getter())


def _resolve_picker_path(value: str, prefix: str, directory_name: str, label: str) -> Path:
    """Resolve a frontend picker token while keeping it inside ComfyUI's directory."""

    raw = str(value).strip()
    if not raw.startswith(prefix):
        return Path(raw).expanduser()

    relative_text = raw[len(prefix) :].replace("\\", "/")
    relative = Path(relative_text)
    if not relative_text or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}选择结果无效：不能使用绝对路径或 ..")

    root = _comfy_directory(directory_name).expanduser().resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label}必须位于 ComfyUI {directory_name}/ 目录内") from exc
    return resolved


def _register_picker_upload_route() -> None:
    """Register the local picker upload endpoint when running inside ComfyUI."""

    try:
        from aiohttp import web
        from server import PromptServer
    except ModuleNotFoundError:
        return

    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None or getattr(prompt_server, "mlx_html_scramble_upload_route", False):
        return

    @prompt_server.routes.post("/mlx_gen/html_scramble_upload")
    async def _upload_picker_file(request):
        post = await request.post()
        uploaded = post.get("file")
        if uploaded is None or not getattr(uploaded, "filename", ""):
            raise web.HTTPBadRequest(text="缺少要上传的文件")

        filename = Path(str(uploaded.filename)).name
        extension = Path(filename).suffix.casefold()
        if extension not in IMAGE_EXTENSIONS | {".html", ".htm"}:
            raise web.HTTPBadRequest(text=f"不支持的文件类型：{extension or '无扩展名'}")

        subfolder = str(post.get("subfolder", "")).strip().replace("\\", "/")
        relative_directory = Path(subfolder)
        if (
            not subfolder
            or relative_directory.is_absolute()
            or ".." in relative_directory.parts
            or relative_directory.parts[:1] != (_PICKER_UPLOAD_ROOT,)
        ):
            raise web.HTTPBadRequest(text="上传目录无效")

        input_root = _comfy_directory("input").expanduser().resolve()
        destination_directory = (input_root / relative_directory).resolve()
        try:
            destination_directory.relative_to(input_root)
        except ValueError as exc:
            raise web.HTTPBadRequest(text="上传目录必须位于 ComfyUI input/ 目录内") from exc

        destination_directory.mkdir(parents=True, exist_ok=True)
        destination = destination_directory / filename
        with destination.open("wb") as output_file:
            shutil.copyfileobj(uploaded.file, output_file)

        relative_path = destination.relative_to(input_root).as_posix()
        return web.json_response({"path": relative_path})

    prompt_server.mlx_html_scramble_upload_route = True


_register_picker_upload_route()


def _strip_downloaded_name_prefix(filename: str) -> str:
    """Remove the downloader prefix from a local filename for matching."""

    return _DOWNLOADED_NAME_PREFIX_RE.sub("", filename, count=1)


def _match_local_image(
    block: _ScrambleBlock,
    spec: dict[str, Any] | None,
    image_paths: list[Path],
) -> Path:
    """Match one HTML block to a downloaded image without making network calls."""

    by_name: dict[str, list[Path]] = {}
    by_stem: dict[str, list[Path]] = {}
    by_stripped_name: dict[str, list[Path]] = {}
    by_stripped_stem: dict[str, list[Path]] = {}
    for path in image_paths:
        by_name.setdefault(path.name.casefold(), []).append(path)
        by_stem.setdefault(path.stem.casefold(), []).append(path)
        stripped_name = _strip_downloaded_name_prefix(path.name)
        stripped_stem = Path(stripped_name).stem
        by_stripped_name.setdefault(stripped_name.casefold(), []).append(path)
        by_stripped_stem.setdefault(stripped_stem.casefold(), []).append(path)

    references: list[str] = []
    if spec is not None:
        references.append(str(spec.get("url", "")))
    references.extend(
        (
            block.element_id,
            block.image_attrs.get("data-original", ""),
            block.image_attrs.get("src", ""),
            block.image_attrs.get("id", ""),
        )
    )
    if block.page is not None:
        references.extend((str(block.page), str(block.page + 1)))

    seen: set[str] = set()
    candidates: list[str] = []
    for reference in references:
        filename = _reference_filename(reference)
        if not filename:
            continue
        for candidate in (filename, Path(filename).stem):
            key = candidate.casefold()
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)

    for candidate in candidates:
        exact = by_name.get(candidate.casefold(), [])
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise ValueError(f"本地图片文件名匹配不唯一：{candidate}")

    for candidate in candidates:
        stripped_matches = by_stripped_name.get(candidate.casefold(), [])
        if len(stripped_matches) == 1:
            return stripped_matches[0]
        if len(stripped_matches) > 1:
            names = ", ".join(path.name for path in stripped_matches)
            raise ValueError(f"去掉 imgi_数字_ 前缀后匹配不唯一：{candidate}（{names}）")

    for candidate in candidates:
        stem_matches = by_stem.get(Path(candidate).stem.casefold(), [])
        if len(stem_matches) == 1:
            return stem_matches[0]
        if len(stem_matches) > 1:
            raise ValueError(f"本地图片主文件名匹配不唯一：{candidate}")

    for candidate in candidates:
        stripped_stem_matches = by_stripped_stem.get(Path(candidate).stem.casefold(), [])
        if len(stripped_stem_matches) == 1:
            return stripped_stem_matches[0]
        if len(stripped_stem_matches) > 1:
            names = ", ".join(path.name for path in stripped_stem_matches)
            raise ValueError(f"去掉 imgi_数字_ 前缀后主文件名匹配不唯一：{candidate}（{names}）")

    references_text = ", ".join(candidates[:6]) or "（HTML 中没有图片文件名）"
    raise FileNotFoundError(
        f"图片目录中找不到与页面 {block.element_id or block.order} 匹配的图片；候选：{references_text}"
    )


def _open_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as source:
            source.load()
            return source.convert("RGBA").copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"无法解析本地图片：{path}（{exc}）") from exc


def _compose_canvas(source: Image.Image, spec: dict[str, Any]) -> Image.Image:
    """Apply the eight-number ``drawImage`` argument arrays from the page."""

    width = _positive_int(spec.get("width"), "canvas width")
    height = _positive_int(spec.get("height"), "canvas height")
    args = spec.get("args")
    if not isinstance(args, list) or not args:
        raise ValueError("canvas data 缺少非空 args 数组")

    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for index, values in enumerate(args):
        if not isinstance(values, list) or len(values) != 8:
            raise ValueError(f"canvas args[{index}] 必须包含 8 个数字")
        try:
            sx, sy, sw, sh, dx, dy, dw, dh = (int(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"canvas args[{index}] 包含非整数值：{values!r}") from exc
        if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
            raise ValueError(f"canvas args[{index}] 的区域尺寸必须大于 0：{values!r}")

        # Browser drawImage clips source rectangles outside the source image.
        # The album data normally stays inside the image, but clipping here
        # makes the node tolerant of a final partial strip.
        left = max(0, sx)
        top = max(0, sy)
        right = min(source.width, sx + sw)
        bottom = min(source.height, sy + sh)
        if right <= left or bottom <= top:
            continue
        piece = source.crop((left, top, right, bottom))
        if (right - left, bottom - top) != (dw, dh):
            piece = piece.resize((dw, dh), Image.Resampling.LANCZOS)

        # Keep the same draw order and alpha behavior as a canvas.
        canvas.paste(piece, (dx, dy), piece)
    return canvas


def _reverse_strips(source: Image.Image, rows: int) -> Image.Image:
    """Fallback for snapshots where the dynamic canvas data was not saved."""

    if rows <= 0:
        return source
    if rows > source.height:
        raise ValueError(f"回退分片数不能大于图片高度：{rows} > {source.height}")
    output = Image.new("RGBA", source.size, (0, 0, 0, 0))
    for destination_row in range(rows):
        source_row = rows - 1 - destination_row
        source_top = source_row * source.height // rows
        source_bottom = (source_row + 1) * source.height // rows
        destination_top = destination_row * source.height // rows
        destination_bottom = (destination_row + 1) * source.height // rows
        piece = source.crop((0, source_top, source.width, source_bottom))
        piece = piece.resize((source.width, destination_bottom - destination_top), Image.Resampling.LANCZOS)
        output.paste(piece, (0, destination_top), piece)
    return output


def _output_stem(block: _ScrambleBlock, source_path: Path, index: int) -> str:
    for value in (
        block.element_id,
        block.image_attrs.get("id", ""),
        source_path.name,
    ):
        stem = Path(value).stem
        if stem:
            return f"{index:04d}_{stem}"
    return f"{index:04d}_image"


def _available_output(path: Path, overwrite: bool) -> Path:
    if overwrite or not path.exists():
        return path
    for suffix in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError(f"无法为已有文件生成新文件名：{path}")


class MlxDownloadScrambledHtmlImages:
    """按 HTML 中的文件名匹配本地图片并按 Canvas 参数恢复为 PNG。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_directory": (
                    "STRING",
                    {
                        "default": "<已下载图片文件夹的绝对路径>",
                        "multiline": False,
                        "tooltip": "已经批量下载好的原始图片目录；节点不会访问网络",
                    },
                ),
                "html_file": (
                    "STRING",
                    {
                        "default": "<HTML 文件的绝对路径>",
                        "multiline": False,
                        "tooltip": "包含 scramble-page 图片 div 的 HTML 文件",
                    },
                ),
                "fallback_scramble_rows": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 200,
                        "tooltip": "canvas data 缺失时按多少条横向条带倒序；0 表示直接保存原图",
                    },
                ),
                "overwrite": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "同名输出是否覆盖；关闭后自动添加 _1、_2 后缀"},
                ),
                "continue_on_error": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "单张匹配或恢复失败时是否跳过并继续处理后面的图片"},
                ),
            }
        }

    RETURN_TYPES = ("images", "STRING", "STRING", "INT")
    RETURN_NAMES = ("images", "saved_files", "report", "success_count")
    OUTPUT_NODE = True
    FUNCTION = "download"
    CATEGORY = "MLX/Gen/Image"

    def download(
        self,
        image_directory: str,
        html_file: str,
        fallback_scramble_rows: int,
        overwrite: bool,
        continue_on_error: bool,
    ):
        image_value = str(image_directory).strip()
        image_root = _resolve_picker_path(
            image_value,
            _PICKER_INPUT_PREFIX,
            "input",
            "图片目录",
        )
        if not image_value or image_root.name.startswith("<"):
            raise ValueError("请把图片目录改成实际的文件夹路径")
        image_paths = _image_paths(image_root)
        output = image_root / "image"
        output.mkdir(parents=True, exist_ok=True)

        html_value = str(html_file).strip()
        html_path = _resolve_picker_path(
            html_value,
            _PICKER_INPUT_PREFIX,
            "input",
            "HTML 文件",
        )
        if not html_value or html_path.name.startswith("<"):
            raise ValueError("请把 HTML 文件改成实际的文件路径")
        if not html_path.is_file():
            raise ValueError(f"HTML 文件不存在：{html_path}")

        try:
            html_text = html_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            html_text = html_path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            raise ValueError(f"无法读取 HTML 文件：{html_path}（{exc}）") from exc

        parser = _AlbumHTMLParser()
        parser.feed(html_text)
        parser.close()
        blocks = _ordered_blocks(parser.blocks)
        if not blocks:
            required = " ".join(sorted(TARGET_CLASSES))
            raise ValueError(f"HTML 中没有找到目标 div（需要 class 同时包含：{required}）")

        restored: list[Image.Image] = []
        saved_paths: list[str] = []
        records: list[dict[str, Any]] = []
        source_path: Path | None = None
        for index, block in enumerate(blocks):
            source_path = None
            spec: dict[str, Any] | None = None
            try:
                spec = _decode_canvas_data(block.canvas_data) if block.canvas_data else None
                source_path = _match_local_image(block, spec, image_paths)
                source = _open_image(source_path)
                if spec is not None:
                    image = _compose_canvas(source, spec)
                    mode = "canvas_args"
                else:
                    image = _reverse_strips(source, int(fallback_scramble_rows))
                    mode = "fallback_reverse_strips" if fallback_scramble_rows else "direct"

                path = _available_output(
                    output / f"{_output_stem(block, source_path, index)}.png",
                    bool(overwrite),
                )
                image.save(path, format="PNG")
                restored.append(image)
                saved_paths.append(str(path))
                records.append(
                    {
                        "index": index,
                        "page": block.page,
                        "source": str(source_path),
                        "output": str(path),
                        "mode": mode,
                        "size": list(image.size),
                        "status": "success",
                    }
                )
                print(
                    f"[MlxDownloadScrambledHtmlImages] [{index + 1}/{len(blocks)}] 已保存：{path}"
                )
            except Exception as exc:  # noqa: BLE001 - report a page and optionally continue.
                record = {
                    "index": index,
                    "page": block.page,
                    "source": str(source_path) if source_path else "",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                records.append(record)
                if not continue_on_error:
                    raise RuntimeError(
                        f"第 {index + 1} 张图片处理失败：{source_path or block.element_id}\n"
                        f"{json.dumps(record, ensure_ascii=False)}"
                    ) from exc
                print(
                    f"[MlxDownloadScrambledHtmlImages] [{index + 1}/{len(blocks)}] 失败，继续：{exc}"
                )

        if not restored:
            report = json.dumps(records, ensure_ascii=False, indent=2)
            raise RuntimeError(f"没有图片处理成功：\n{report}")

        report = json.dumps(records, ensure_ascii=False, indent=2)
        print(
            f"[MlxDownloadScrambledHtmlImages] 完成：成功 {len(restored)}/{len(blocks)}，输出目录：{output}"
        )
        return MlxPilImage(images=tuple(restored)), "\n".join(saved_paths), report, len(restored)


__all__ = ["MlxDownloadScrambledHtmlImages"]