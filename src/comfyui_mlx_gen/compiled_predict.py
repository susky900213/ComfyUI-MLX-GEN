"""缓存 `mx.compile` 之后的单步预测函数（同一函数 + 同一形状只 trace 一次）。

对齐参考实现 mflux `mflux/utils/compiled_predict_cache.py`（0.25.0 起，
`Flux2Initializer._init_config` 会给 model 挂一个实例）。语义要点：

- `mx.compile` 会把闭包里的权重数组当常量烘进计算图 → 缓存的 callable 只在
  「闭包住的模块还是同一个、且权重没被换掉」时有效，所以用 `weights_token`
  记录模块身份：模块被换掉（换权重目录 / 重新加载 / 释放后重建）时整桶丢掉，
  否则旧计算图会把旧权重一直扣住。
- 入参的 shape/dtype 变化由 `mx.compile` 自己在内部按需重新 trace，**不进键**；
  键只覆盖 **Python 层的分支结构**（如有没有 negative 条件、edit 走 extract
  还是 cached 分支）。
- 在同一个模块上**就地**改过权重（挂 / 卸 LoRA、重刷量化）必须显式 `clear()`
  —— 模块身份查不出这种事（本插件目前只在物化组件时改权重，改完才进缓存，
  因此没有这种站点；`clear()` 保留给将来用）。
"""

from __future__ import annotations

from collections.abc import Callable, Hashable


class CompiledPredictCache:
    """以「分支结构」为键缓存 compiled（或 eager）predict callable。"""

    def __init__(self) -> None:
        self._entries: dict[Hashable, Callable] = {}
        self._weights_token: object | None = None

    def get_or_build(self, *, key: Hashable, weights_token: object, build: Callable[[], Callable]) -> Callable:
        if weights_token is not self._weights_token:
            # 模块被换掉（释放 / 重载）：缓存里的计算图都闭住了旧数组，
            # 全部丢掉才不会把旧权重一直扣在显存里。
            self._entries = {}
            self._weights_token = weights_token
        compiled = self._entries.get(key)
        if compiled is None:
            compiled = build()
            self._entries[key] = compiled
        return compiled

    def clear(self) -> None:
        self._entries = {}
        self._weights_token = None

    def __len__(self) -> int:
        return len(self._entries)