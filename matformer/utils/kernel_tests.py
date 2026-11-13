"""Helpers for the `_test()` of the registered kernels.

A kernel can be installed on a machine where it cannot run (no GPU, a wheel built for another
torch/CUDA version, a triton kernel without triton support...), and that only shows up when the
kernel is really called. The registry runs `_test()` once per process, at instantiation time, and
skips the implementation for good when it fails, so these helpers stay deliberately minimal: build
the kernel, run it on the smallest possible input, let the exception travel to the registry.
"""
from matformer.dataclasses.tensors_dataclasses import NormalTensor


def kernel_runs(build, tensor):
    """Build a kernel with `build()` and run it once on a TensorDC wrapping `tensor`."""
    build()(NormalTensor(tensor=tensor))
    return True
