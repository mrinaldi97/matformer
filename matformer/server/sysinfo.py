# matformer/server/sysinfo.py
"""Machine status and framework introspection: what the box has (GPUs, RAM, versions) and what
the registry knows (categories, variants, implementations)."""
import importlib.metadata
import os
import platform
import shutil
import socket
import subprocess
import sys

from matformer.server import config
from matformer.server.core import rpc


def _meminfo() -> dict:
    out = {}
    for line in open("/proc/meminfo"):
        key, _, rest = line.partition(":")
        out[key] = int(rest.split()[0]) * 1024
    return out


def _gpus() -> list:
    """nvidia-smi, without ever importing torch: the GUI shows the box, not the interpreter."""
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,driver_version",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if query.returncode != 0:
            return []
        return [{"index": int(i), "name": n.strip(), "mem_used_mib": int(u), "mem_total_mib": int(t),
                 "util": int(x), "driver": d.strip()}
                for i, n, u, t, x, d in (line.split(",") for line in query.stdout.strip().splitlines())]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


@rpc("sys.info")
def sys_info(conn, p):
    """Hostname, OS, python, torch/CUDA versions (metadata, no imports), GPUs, RAM, disk."""
    mem = _meminfo() if sys.platform.startswith("linux") else {}
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    cfg = config()
    du = shutil.disk_usage(cfg["roots"][0])
    return {"name": cfg["name"], "host": socket.gethostname(), "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(), "torch": torch_version,
            "cuda": os.environ.get("CUDA_VERSION") or "see driver", "pid": os.getpid(),
            "cwd": os.getcwd(), "roots": cfg["roots"],
            "ram_total": mem.get("MemTotal"), "ram_free": mem.get("MemAvailable"),
            "disk_free": du.free, "disk_total": du.total, "gpus": _gpus()}


@rpc("sys.mem")
def sys_mem(conn, p):
    """Memory of the server process: RSS plus, when torch is loaded, the CUDA allocator."""
    rss = None
    try:
        for line in open(f"/proc/{os.getpid()}/status"):
            if line.startswith("VmRSS"):
                rss = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    out = {"ram_used": rss}
    import sys
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
        torch = sys.modules["torch"]
        out.update(cuda_allocated=torch.cuda.memory_allocated(), cuda_reserved=torch.cuda.memory_reserved(),
                   cuda_total=torch.cuda.get_device_properties(0).total_memory)
    return out


@rpc("sys.gc")
def sys_gc(conn, p):
    """Collect and release the CUDA cache back to the driver."""
    import gc
    before = sys_mem(conn, p).get("cuda_reserved")
    gc.collect()
    import sys
    if "torch" in sys.modules:
        sys.modules["torch"].cuda.empty_cache()
        sys.modules["torch"].cuda.synchronize()
    after = sys_mem(conn, p).get("cuda_reserved")
    freed = (before - after) if isinstance(before, int) and isinstance(after, int) else None
    return {"msg": f"freed {freed / 2**20:.0f} MiB" if freed else "collected"}


@rpc("reg.categories")
def reg_categories(conn, p):
    """Every registry category with its default variant and the list of variants."""
    from matformer.matformer_registry import registry
    info = registry.get_registry_info(include_implementations=False)
    return [{"category": name, "default": cat["default_variant"], "variants": sorted(cat["variants"])}
            for name, cat in sorted(info.items())]


@rpc("reg.info")
def reg_info(conn, p):
    """Full registry tree for one category (variants, definitions, implementations, priorities)."""
    from matformer.matformer_registry import registry
    return registry.get_registry_info(category=p.get("category"), variant=p.get("variant"),
                                      relative_filepath=True)
