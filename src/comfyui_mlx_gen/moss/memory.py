"""Small persistent memory store for the MOSS prompt enhancer.

The store deliberately keeps only text.  Reference images are sent to the
model for the current request but are never persisted in the memory file.
Each ComfyUI node gets an independent namespace based on its ``UNIQUE_ID``.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


MAX_CONTEXT_TURNS = 20
MAX_MEMORY_ITEMS = 20
_STORE_VERSION = 1


def default_memory_path() -> Path:
    """Return the user-local memory file path, overridable for deployments/tests."""

    configured = str(os.environ.get("COMFYUI_MLX_GEN_MOSS_MEMORY_FILE", "")).strip()
    if configured:
        return Path(configured).expanduser()

    state_directory = str(os.environ.get("COMFYUI_MLX_GEN_STATE_DIR", "")).strip()
    if state_directory:
        return Path(state_directory).expanduser() / "moss_memory.json"

    # ComfyUI exposes its user directory in normal server processes.  Keep the
    # import optional so the node remains testable without a ComfyUI checkout.
    try:
        import folder_paths

        user_directory = getattr(folder_paths, "get_user_directory", None)
        if callable(user_directory):
            return Path(user_directory()) / "comfyui_mlx_gen_moss_memory.json"
    except (ImportError, OSError, TypeError):
        pass

    return Path.home() / ".comfyui_mlx_gen" / "moss_memory.json"


def _text(value: Any, *, limit: int = 8000) -> str:
    return str(value or "").strip()[:limit]


def _node_key(node_id: Any) -> str:
    value = _text(node_id, limit=200)
    return value or "__unscoped__"


def _empty_data() -> dict[str, Any]:
    return {"version": _STORE_VERSION, "contexts": {}, "memories": {}}


class MossMemoryStore:
    """Thread-safe JSON-backed context and long-term memory for MOSS nodes."""

    def __init__(
        self,
        path: str | os.PathLike[str] | Path | None = None,
        *,
        max_context_turns: int = MAX_CONTEXT_TURNS,
        max_memory_items: int = MAX_MEMORY_ITEMS,
    ) -> None:
        self.path = Path(path) if path is not None else default_memory_path()
        self.max_context_turns = max(1, int(max_context_turns))
        self.max_memory_items = max(1, int(max_memory_items))
        self._lock = threading.RLock()

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return _empty_data()
        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            # A corrupt optional memory file must not prevent prompt execution.
            return _empty_data()
        if not isinstance(parsed, dict):
            return _empty_data()
        data = _empty_data()
        data["contexts"] = parsed.get("contexts") if isinstance(parsed.get("contexts"), dict) else {}
        data["memories"] = parsed.get("memories") if isinstance(parsed.get("memories"), dict) else {}
        return data

    def _write_unlocked(self, data: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary = Path(output.name)
                json.dump(data, output, ensure_ascii=False, indent=2)
                output.write("\n")
            temporary.replace(self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink(missing_ok=True)

    def get_context(self, node_id: Any, limit: int = MAX_CONTEXT_TURNS) -> list[dict[str, Any]]:
        """Return the newest valid context turns for one node."""

        key = _node_key(node_id)
        count = max(0, min(int(limit), self.max_context_turns))
        if count == 0:
            return []
        with self._lock:
            values = self._read_unlocked()["contexts"].get(key, [])
            if not isinstance(values, list):
                return []
            return [dict(item) for item in values[-count:] if isinstance(item, Mapping)]

    def get_memories(self, node_id: Any) -> list[dict[str, Any]]:
        """Return all valid long-term memories for one node."""

        key = _node_key(node_id)
        with self._lock:
            values = self._read_unlocked()["memories"].get(key, [])
            if not isinstance(values, list):
                return []
            return [dict(item) for item in values if isinstance(item, Mapping)]

    def record_turn(
        self,
        node_id: Any,
        *,
        prompt: str,
        enhanced_prompt: str,
        mode: str,
        has_image: bool,
        save_memory: bool,
    ) -> None:
        """Record one successful turn and optionally add it to long-term memory."""

        request_text = _text(prompt)
        result_text = _text(enhanced_prompt)
        if not request_text or not result_text:
            return
        key = _node_key(node_id)
        turn = {
            "prompt": request_text,
            "enhanced_prompt": result_text,
            "mode": _text(mode, limit=100),
            "has_image": bool(has_image),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            data = self._read_unlocked()
            contexts = data["contexts"].setdefault(key, [])
            if not isinstance(contexts, list):
                contexts = data["contexts"][key] = []
            contexts.append(turn)
            del contexts[:-self.max_context_turns]

            if save_memory:
                memories = data["memories"].setdefault(key, [])
                if not isinstance(memories, list):
                    memories = data["memories"][key] = []
                memories.append(dict(turn))
                del memories[:-self.max_memory_items]
            self._write_unlocked(data)

    def clear(self, node_id: Any, kind: str = "all") -> None:
        """Clear ``context``, ``memory``, or ``all`` for one node."""

        if kind not in {"context", "memory", "all"}:
            raise ValueError(f"未知 MOSS 记忆清除类型：{kind}")
        key = _node_key(node_id)
        with self._lock:
            data = self._read_unlocked()
            if kind in {"context", "all"}:
                data["contexts"].pop(key, None)
            if kind in {"memory", "all"}:
                data["memories"].pop(key, None)
            self._write_unlocked(data)


def register_memory_routes(store: MossMemoryStore) -> None:
    """Register the frontend buttons' clear endpoint when running in ComfyUI."""

    try:
        from aiohttp import web
        from server import PromptServer
    except ModuleNotFoundError:
        return

    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None or getattr(prompt_server, "mlx_moss_memory_route", False):
        return

    @prompt_server.routes.post("/mlx_gen/moss_memory/clear")
    async def _clear_memory(request):
        try:
            payload = await request.json()
        except (TypeError, ValueError):
            raise web.HTTPBadRequest(text="MOSS 记忆清除请求不是有效 JSON")
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(text="MOSS 记忆清除请求必须是对象")
        node_id = _text(payload.get("node_id"), limit=200)
        kind = _text(payload.get("kind"), limit=20)
        if not node_id:
            raise web.HTTPBadRequest(text="缺少 MOSS 节点 ID")
        try:
            store.clear(node_id, kind)
        except ValueError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        return web.json_response({"ok": True, "node_id": node_id, "kind": kind})

    prompt_server.mlx_moss_memory_route = True


__all__ = ["MossMemoryStore", "MAX_CONTEXT_TURNS", "MAX_MEMORY_ITEMS", "default_memory_path", "register_memory_routes"]