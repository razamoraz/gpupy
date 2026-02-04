import time
import contextlib
import cupy as cp
import warp as wp
import numpy as np

class Profiler:
    def __init__(self, name="Benchmark"):
        self.name = name
        self.start_time = 0
        self.end_time = 0
        self.duration = 0
    
    def __enter__(self):
        # Synchronize GPU to ensure start time is accurate for GPU tasks
        cp.cuda.Stream.null.synchronize()
        wp.synchronize()
        cp.cuda.nvtx.RangePush(self.name)
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Synchronize GPU to ensure end time is accurate
        cp.cuda.Stream.null.synchronize()
        wp.synchronize()
        self.end_time = time.perf_counter()
        cp.cuda.nvtx.RangePop()
        self.duration = self.end_time - self.start_time

@contextlib.contextmanager
def profile_section(name):
    cp.cuda.nvtx.RangePush(name)
    try:
        yield
    finally:
        cp.cuda.nvtx.RangePop()
