import time


class RuntimeTimer:
    def __init__(self):
        self._start = time.perf_counter()

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._start
