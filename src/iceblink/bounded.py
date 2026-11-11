"""Both queued and in-flight observations count against the byte budget."""
from queue import Queue, Empty
import threading


class BudgetQueue:
    def __init__(self, count, bytes_limit):
        self._queue = Queue(maxsize=count)
        self._lock = threading.Lock()
        self.limit_count, self.limit_bytes = count, bytes_limit
        self.count = self.bytes = self.peak_bytes = 0

    def put(self, item, size):
        with self._lock:
            if size < 0 or self.count >= self.limit_count or self.bytes + size > self.limit_bytes:
                return False
            self.count += 1
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
            self._queue.put_nowait((item, size))
            return True

    def get(self, timeout=None):
        return self._queue.get(timeout=timeout)

    def release(self, size):
        with self._lock:
            self.count -= 1
            self.bytes -= size
            if self.count < 0 or self.bytes < 0:
                raise RuntimeError("Queue budget released twice")
        self._queue.task_done()
