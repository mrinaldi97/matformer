#matformer/utils/ckpt_manager.py
import csv
from datetime import datetime
from pathlib import Path


class CheckpointDirectoryManager:
    """
    Manages checkpoint filenames, versioning, and a lightweight resume log.

    Naming convention:
        {save_path}/{model_name}_v{version}_s{step}.ckpt

    Resume log (matformer_resume.csv):
        model_name, version, step, filename, timestamp

 
    """

    version_string = "_v_"
    step_string    = "_s_"
    extension      = ".ckpt"
    resume_file    = "matformer_resume.csv"

    def __init__(self, save_path: str, model_name: str, version: int = None):
        self.save_path    = Path(save_path)
        self.model_name   = model_name
        self.current_step = 0

        self.save_path.mkdir(parents=True, exist_ok=True)

        log = self.save_path / self.resume_file
        if not log.exists():
            log.write_text("model_name,version,step,filename,timestamp\n")

        self.current_version = version if version is not None else self._latest_version()


    def _scan(self) -> list[tuple[int, int, Path]]:
        """Parse all on-disk checkpoints → [(version, step, path)]."""
        out, prefix = [], self.model_name + self.version_string
        for p in self.save_path.glob(f"{prefix}*{self.extension}"):
            try:
                v_part      = p.stem.split(self.version_string)[1]   # "2_s5000"
                v, s        = v_part.split(self.step_string)
                out.append((int(v), int(s), p))
            except (IndexError, ValueError):
                continue
        return out

    def _latest_version(self) -> int:
        return max((v for v, _, _ in self._scan()), default=1)

    def _ckpt_path(self, step: int) -> Path:
        name = f"{self.model_name}{self.version_string}{self.current_version}{self.step_string}{step}{self.extension}"
        return self.save_path / name

    def _version_ckpts(self) -> list[tuple[int, Path]]:
        return [(s, p) for v, s, p in self._scan() if v == self.current_version]


    def set_version(self, version: int):
        self.current_version = version

    def advance_version(self):
        self.current_version += 1
        self.current_step = 0

    def write_resume(self, step: int, path: Path):
        with open(self.save_path / self.resume_file, "a", newline="") as f:
            csv.writer(f).writerow([
                self.model_name, self.current_version, step,
                path.name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ])

    def return_save_path(self, step: int) -> Path:
        """Call this instead of torch.save to get the right path and log it."""
        self.current_step = step
        path = self._ckpt_path(step)
        self.write_resume(step, path)
        return path


    def resume(self) -> tuple[Path | None, int]:
        """Last checkpoint of current version, or (None, 0) if none exists."""
        ckpts = self._version_ckpts()
        if not ckpts:
            return None, 0
        step, path = max(ckpts, key=lambda x: x[0])
        self.current_step = step
        return path, step

    def start_from_scratch(self) -> tuple[None, int]:
        """Advance version, reset step. Load path is None (no weights to load)."""
        self.advance_version()
        return None, 0

    def restart_from_step(self, desired_step: int) -> tuple[Path | None, int]:
        """Closest checkpoint >= desired_step in current version, then advance version."""
        ckpts = self._version_ckpts()
        above = [(s, p) for s, p in ckpts if s >= desired_step]
        step, path = min(above or ckpts, key=lambda x: x[0]) if ckpts else (0, None)
        self.advance_version()
        self.current_step = step
        return path, step
