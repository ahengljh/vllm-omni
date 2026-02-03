# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext

import torch
from vllm.logger import init_logger

from .base import ProfilerBase

logger = init_logger(__name__)


def _is_cuda_available() -> bool:
    """Check if CUDA is available on the current platform."""
    return torch.cuda.is_available()


class CudaProfiler(ProfilerBase):
    """
    Lightweight profiler that signals nsys via the CUDA Profiler API.

    When the server is launched under ``nsys profile
    --capture-range=cudaProfilerApi``, calling ``start()`` /
    ``stop()`` brackets the region that nsys will capture.  No trace
    files are written by this class — nsys handles all tracing
    externally and produces a ``.nsys-rep`` file on process exit.

    On non-CUDA platforms (ROCm, NPU, XPU, etc.) all operations are
    no-ops and a warning is logged on the first call.
    """

    _active: bool = False
    _warned_no_cuda: bool = False

    @classmethod
    def _warn_no_cuda(cls) -> None:
        if not cls._warned_no_cuda:
            logger.warning(
                "CudaProfiler requires CUDA but current platform does "
                "not support it. Profiler calls will be no-ops."
            )
            cls._warned_no_cuda = True

    @classmethod
    def start(cls, trace_path_template: str = "") -> str:
        """Start the CUDA profiler range for nsys capture."""
        if not _is_cuda_available():
            cls._warn_no_cuda()
            return ""
        if cls._active:
            logger.warning("[Rank %s] CUDA profiler already active", cls._get_rank())
            return ""
        torch.cuda.profiler.start()
        cls._active = True
        logger.info("[Rank %s] CUDA profiler started (nsys capture region open)", cls._get_rank())
        return ""

    @classmethod
    def stop(cls) -> str | None:
        """Stop the CUDA profiler range for nsys capture."""
        if not _is_cuda_available():
            cls._warn_no_cuda()
            return None
        if not cls._active:
            return None
        torch.cuda.profiler.stop()
        cls._active = False
        logger.info("[Rank %s] CUDA profiler stopped (nsys capture region closed)", cls._get_rank())
        return None

    @classmethod
    def get_step_context(cls):
        """Return an NVTX range context manager when active, else no-op."""
        if cls._active and _is_cuda_available():
            return torch.cuda.nvtx.range("step")
        return nullcontext()

    @classmethod
    def is_active(cls) -> bool:
        return cls._active
