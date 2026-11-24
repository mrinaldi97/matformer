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
        self.save_path = Path(save_path)
        self.model_name = model_name
        self.current_step = 0
        self.current_version = version if version is not None else self._latest_version()


    def _scan(self) -> list[tuple[int, int, Path]]:
        """Parse all on-disk checkpoints -> [(version, step, path)]."""
        if not self.save_path.is_dir():
            return []
        out, prefix = [], self.model_name + self.version_string
        for p in self.save_path.glob(f"{prefix}*{self.extension}"):
            if (parsed := self.parse(p.name)):
                out.append((parsed[1], parsed[2], p))
        return out

    @classmethod
    def parse(cls, filename: str):
        """The (model_name, version, step) one checkpoint filename states, None when it is not ours.

        The filename is the whole of the resume metadata, so the scan above and the selection below
        both read it through here: one place knows the convention."""
        try:
            name, _, tail = Path(filename).stem.partition(cls.version_string)
            version, _, step = tail.partition(cls.step_string)
            return name, int(version), int(step)
        except ValueError:
            return None

    def _latest_version(self) -> int:
        return max((v for v, _, _ in self._scan()), default=1)

    @classmethod
    def select(cls, location, model_name: str = None, every_n_steps: int = 0,
               versions=None, last: int = 0) -> list[str]:
        """The checkpoints one location stands for, oldest first.

        `location` may be a single checkpoint, a HuggingFace repo, a loose folder... anything the
        loaders already read: when it does not hold Matformer checkpoints it comes back as it is,
        so a selection rule never swallows a plain path. When it does, the rule picks among what is
        there -- `model_name` one run of them, `every_n_steps` the steps that are a multiple of it,
        only the newest version per run unless `versions` asks for others, and `last` the N newest
        of what is left.

        This is what lets a config address a *series* of models ("every 500 steps of this run")
        instead of one file, and so lets a sweep be declared over models as plainly as over
        learning rates."""
        path = Path(location)
        if not path.is_dir():
            return [str(location)]
        found = []
        for one in sorted(path.glob(f"*{cls.extension}")):
            parsed = cls.parse(one.name)
            if parsed and (model_name is None or parsed[0] == model_name):
                found.append(parsed + (one,))
        if not found:
            return [str(location)]               # not a checkpoint folder: the loaders know it best
        found.sort(key=lambda one: (one[0], one[1], one[2]))     # numeric, not by filename
        if versions:
            found = [one for one in found if one[1] in set(versions)]
        else:                              # each run contributes its newest version, and one only
            newest = {}
            for name, version, *_ in found:
                newest[name] = max(newest.get(name, version), version)
            found = [one for one in found if one[1] == newest[one[0]]]
        if every_n_steps:
            stepped = [one for one in found if one[2] % int(every_n_steps) == 0]
            if not stepped:
                raise ValueError(f"every_n_steps={every_n_steps} selects none of the checkpoints "
                                 f"in {path} (steps: "
                                 f"{', '.join(str(one[2]) for one in found[:8])})")
            found = stepped
        picked = [str(one[3]) for one in found]
        return picked[-int(last):] if (last and picked) else (picked or [str(location)])

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

    def return_save_path(self, step: int) -> Path:
        self.current_step = step
        self.save_path.mkdir(parents=True, exist_ok=True)
        return self._ckpt_path(step)
    def mark_saved(self, step: int, path: Path):
        """Append to the resume log. Call on rank 0 only, after the save succeeded."""
        log = self.save_path / self.resume_file
        is_new = not log.exists()
        with open(log, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["model_name", "version", "step", "filename", "timestamp"])
            writer.writerow([self.model_name, self.current_version, step,
                            path.name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

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
    def plugins_dir(self, step: int) -> Path:
        """Collision-free folder for one step's plugin checkpoints, next to the main ckpt."""
        d = self.save_path / f"{self.model_name}{self.version_string}{self.current_version}{self.step_string}{step}_plugins"
        d.mkdir(parents=True, exist_ok=True)
        return d
    def decide_checkpoint(self, checkpoint=None, start_from_scratch=False, start_from_step=None):
        """
        Decide what this run starts from. Returns (load_path, step, reason).
  
        - checkpoint: explicit Matformer .ckpt. Continues that run's state (optimizer, scheduler,
        datamodule, step) in a new version; to fine-tune from it, set what_to_reset to "optimizer,lr_scheduler,datamodule,steps".
        - start_from_scratch: new version, nothing loaded.
        - start_from_step: closest checkpoint >= that step in the current version, then new version.
        - otherwise: resume the latest checkpoint of the current version, or scratch if none.
        """
        if checkpoint:
            if not Path(checkpoint).is_file():
                raise FileNotFoundError(f"Explicit checkpoint not found: {checkpoint}")
            self.advance_version()
            return Path(checkpoint), 0, f"explicit checkpoint {checkpoint}"
        if start_from_scratch:
            path, step = self.start_from_scratch()
            return path, step, "From scratch (new version)"
        path, step = (self.restart_from_step(start_from_step) if start_from_step
                    else self.resume())
        reason = f"{path} at step {step}" if path else "From scratch (nothing to resume)"
        return path, step, reason        