"""Runtime paths for relocated V3 code, shards and original source files."""
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath


DEFAULT_ROOT = Path(os.environ.get(
    "V3_DATA_ROOT", Path(__file__).resolve().parents[1] / "adaptive_resampling_data_v3"
))


class SourcePaths:
    """Map recorded raw-data paths without rewriting manifests or HDF5 shards.

    V3_RAW_ROOT is the new location of generation_config.json's raw_root.
    Both Windows separators and POSIX separators are accepted in stored paths.
    Relative source paths are interpreted relative to the V3 dataset directory.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.raw_root = os.environ.get("V3_RAW_ROOT")
        config_path = self.root / "metadata/generation_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        self.recorded_root = str(config.get("raw_root", "")).replace("\\", "/").rstrip("/")
        if self.raw_root and not self.recorded_root:
            raise ValueError("V3_RAW_ROOT requires raw_root in metadata/generation_config.json")

    def resolve(self, value):
        normalized = str(value).replace("\\", "/")
        if self.raw_root:
            prefix = self.recorded_root + "/"
            # Windows drive paths are case-insensitive; POSIX paths are not.
            windows = bool(PureWindowsPath(self.recorded_root).drive)
            candidate = normalized.casefold() if windows else normalized
            match_prefix = prefix.casefold() if windows else prefix
            if candidate.startswith(match_prefix):
                return str(Path(self.raw_root).joinpath(*normalized[len(prefix):].split("/")))
        if PurePosixPath(normalized).is_absolute() or PureWindowsPath(normalized).is_absolute():
            return normalized
        return str(self.root / normalized)
