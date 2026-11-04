"""Run the ours1 stages sequentially, continuing completed or interrupted runs."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[1]
ENTRY = REPO / "our1/PUFM-main/adaptive_v3.py"
VERSION = "ours1-rgdb-relflow-v1"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def stage_action(last, best, stage, epochs, data_hash):
    if last is None:
        if best is not None:
            raise ValueError(f"{stage}: best.pt exists but last.pt is missing")
        return "start"
    config = last.get("configuration", {})
    if (last.get("network_version") != VERSION or last.get("protocol") != "V3" or
            last.get("partial_debug_run", True) or last.get("stage") != stage or
            config.get("allocation") != "gdb" or config.get("relation", False) != (stage == "joint") or
            last.get("args", {}).get("epochs") != epochs):
        raise ValueError(f"{stage}: existing checkpoint does not match this pipeline")
    completed = last["epoch"] + 1
    if not 0 < completed <= epochs:
        raise ValueError(f"{stage}: invalid completed epoch count")
    if best is not None:
        selection = best.get("best_validation") or {}
        protocol = selection.get("protocol", {})
        if (best.get("configuration") != config or best.get("stage") != stage or
                best.get("partial_debug_run", True) or not protocol.get("full_validation") or
                protocol.get("data_manifest_sha256") != data_hash):
            raise ValueError(f"{stage}: best.pt is incompatible or selected on partial/different validation data")
    if completed < epochs:
        return "resume"
    if best is None:
        return "select"
    return "done" if best.get("selection_completed_epochs") == epochs else "resume"


def option(command, flag):
    for index, item in enumerate(command):
        if item == flag and index + 1 < len(command):
            return command[index + 1]
        if item.startswith(flag + "="):
            return item.split("=", 1)[1]
    return None


def active_workers(output=None):
    """Only identify workers whose entry and resolved output match this run."""
    result = []
    proc = Path("/proc")
    if not proc.is_dir():
        return result
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            args = directory.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
            cwd = directory.joinpath("cwd").resolve(strict=True)
            if output is None:
                entries = [(cwd / item).resolve() for item in args if item.endswith(".py")]
                is_entry = any(REPO in path.parents and path != Path(__file__).resolve() for path in entries)
                is_entry = is_entry and any(item in {"train", "train-gain"} for item in args)
            else:
                is_entry = any((cwd / item).resolve() == ENTRY for item in args if item.endswith("adaptive_v3.py"))
            target = option(args, "--output")
            if is_entry and (output is None or target and (cwd / target).resolve() == output.resolve()):
                result.append(int(directory.name))
        except (OSError, UnicodeError):
            continue
    return result


def wait_for_workers(output):
    while True:
        workers = active_workers(output)
        if not workers:
            return
        print(f"Waiting for existing project training workers {workers}; output={output}", flush=True)
        time.sleep(30)


@contextmanager
def pipeline_lock(folder):
    if os.name != "posix":
        raise RuntimeError("Run this pipeline on the Linux training server")
    import fcntl
    with (folder / "pipeline.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another pipeline is already running in this output directory")
        yield


def run_command(args, label, command):
    logfile = args.output / label / "console.log" if label in {"rgr", "gain", "joint"} else args.output / f"{label}.console.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    print(f"START {label}; log={logfile}", flush=True)
    with logfile.open("a", encoding="utf-8") as handle:
        handle.write("\n" + time.strftime("%Y-%m-%d %H:%M:%S") + " " + repr(command) + "\n")
        handle.flush()
        with subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(line, end="", flush=True)
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
    print(f"DONE {label}", flush=True)


def train_stage(args, stage, epochs, parent):
    directory = args.output / stage
    wait_for_workers(directory)
    last_path, best_path = directory / "last.pt", directory / "best.pt"
    last = load(last_path) if last_path.exists() else None
    best = load(best_path) if best_path.exists() else None
    action = stage_action(last, best, stage, epochs, args.data_hash)
    if last is not None:
        if last.get("training_profile") != args.training_profile:
            raise ValueError(f"{stage}: training data profile differs from the common checkpoint")
        if stage in {"rgr", "joint"} and last.get("args", {}).get("batch_size", 8) != 8:
            raise ValueError(f"{stage}: checkpoint used batch={last['args']['batch_size']}; this pipeline uses batch=8. Start this stage in a new output from its parent best.pt.")
        # A stage can only continue the parent chosen for this run.
        previous_parent = last.get("args", {}).get("checkpoint")
        if previous_parent and (REPO / previous_parent).resolve() != parent.resolve():
            raise ValueError(f"{stage}: saved parent {previous_parent} differs from {parent}")
    inputs_path = directory / "pipeline_inputs.json"
    expected_inputs = dict(parent_sha256=digest(parent), data_sha256=args.data_hash, source_sha256=args.source_hash)
    if inputs_path.exists() and json.loads(inputs_path.read_text()) != expected_inputs:
        raise ValueError(f"{stage}: parent weights, data, or generator code changed since this stage began")
    if last is not None and stage != "rgr" and not inputs_path.exists():
        raise ValueError(f"{stage}: existing manual stage has no pipeline provenance; use a separate output")
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(inputs_path, expected_inputs)
    command = [sys.executable, "-u", "-B", str(ENTRY)]
    common = ["--root", str(args.root), "--device", "cuda", "--seed", "21", "--output", str(directory)]
    if action == "done":
        print(f"SKIP completed {stage}; best={best_path}", flush=True)
        return best_path
    if action == "select":
        run_command(args, stage + "_select", command + ["select-best", "--checkpoint", str(last_path), *common])
        return best_path
    initial = ["--resume", str(last_path)] if action == "resume" else ["--checkpoint", str(parent)]
    validation = ["--val-every", "5", "--val-objects", "0"]
    assignment_workers = last.get("args", {}).get("assignment_workers", 8) if last else 8
    common += ["--assignment-workers", str(assignment_workers)]
    if stage == "gain":
        training = ["train-gain", "--teacher", str(args.output / "teacher.npz"), "--lr", "1e-3"]
    else:
        training = ["train", "--stage", stage, "--preset", "gdb" if stage == "rgr" else "gdb_flow",
                    "--batch-size", "8", "--workers", "0",
                    "--training-mode", "mixed", "--train-patches-per-object", "32",
                    "--lr", "1e-4" if stage == "rgr" else "1e-5"]
        if stage == "joint" and action == "start":
            training += ["--fm-endpoint", "uniform", "--geom-weight", "1", "--cover-weight", "1", "--retain-weight", "0"]
    if last is None or last.get("args", {}).get("fused_optimizer"):
        training.append("--fused-optimizer")
    del last, best
    run_command(args, stage, command + training + initial + validation + ["--epochs", str(epochs), *common])
    # Inspect after a successful child exit, including final best-selection stamping.
    if stage_action(load(last_path), load(best_path), stage, epochs, args.data_hash) != "done":
        raise RuntimeError(f"{stage} exited without complete training and best selection")
    return best_path


def build_teacher(args, rgr):
    cache = args.output / "teacher.npz"
    metadata_path = cache.with_suffix(".json")
    marker_path = args.output / "teacher.pipeline.json"
    wait_for_workers(cache)
    expected = dict(parent_sha256=digest(rgr), source_sha256=args.source_hash)
    marker = json.loads(marker_path.read_text()) if marker_path.exists() else {}
    if (cache.exists() and metadata_path.exists() and marker.get("inputs") == expected and
            marker.get("cache_sha256") == digest(cache) and marker.get("metadata_sha256") == digest(metadata_path)):
        print("SKIP completed teacher", flush=True)
        return
    if (args.output / "gain/last.pt").exists():
        raise ValueError("Existing GainNet has an unverified teacher cache; use its original pipeline/cache")
    if cache.exists() or metadata_path.exists():
        # Keep manually created or interrupted artifacts available for inspection.
        suffix = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        for path in (cache, metadata_path):
            if path.exists():
                path.rename(path.with_name(path.name + ".before_pipeline_" + suffix))
    run_command(args, "teacher", [sys.executable, "-u", "-B", str(ENTRY), "build-teacher",
        "--root", str(args.root), "--device", "cuda", "--checkpoint", str(rgr),
        "--samples", "128", "--pairs", "4", "--teacher-epochs", "0", "1", "2", "3", "4", "--output", str(cache)])
    marker = dict(inputs=expected, cache_sha256=digest(cache), metadata_sha256=digest(metadata_path))
    atomic_json(marker_path, marker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "adaptive_resampling_data_v3")
    parser.add_argument("--output", type=Path, default=REPO / "server_runs/patch32/ours1_gdb_flow")
    parser.add_argument("--common", type=Path, default=REPO / "server_runs/patch32/pufm_common_b8_w0_a8_fused/epoch_0060.pt")
    args = parser.parse_args()
    args.root, args.output, args.common = (path.resolve() for path in (args.root, args.output, args.common))
    args.output.mkdir(parents=True, exist_ok=True)
    args.data_hash = digest(args.root / "metadata/base_manifest.json")
    sources = [REPO / "our1/PUFM-main/gain_teacher.py", *(REPO / "our1/PUFM-main/models").rglob("*.py")]
    args.source_hash = hashlib.sha256(b"".join(path.read_bytes().replace(b"\r\n", b"\n")
        for path in sorted(sources))).hexdigest()
    os.environ["V3_CODE_ROOT"], os.environ["V3_DATA_ROOT"] = str(REPO), str(args.root)
    os.environ["PYTHONIOENCODING"] = "utf-8"
    with pipeline_lock(args.output):
        wait_for_workers(None)
        common = load(args.common)
        if (common.get("protocol") != "V3" or common.get("stage") != "baseline" or
                common.get("epoch") != 59 or common.get("partial_debug_run", True)):
            raise ValueError("Expected the completed, non-debug common PUFM 60-epoch checkpoint")
        sys.path.insert(0, str(REPO))
        from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset
        from adaptive_resampling.v3_training import require_training_profile
        dataset = AdaptiveResamplingDataset(args.root, "train", include_supervision=False, train_patches_per_object=32)
        args.training_profile = dataset.training_profile
        dataset.close()
        require_training_profile(common, args.training_profile, context="Pipeline common checkpoint")
        del common
        rgr = train_stage(args, "rgr", 20, args.common)
        build_teacher(args, rgr)
        gain = train_stage(args, "gain", 60, rgr)
        for condition in ("A0_normal", "C2_hole_10", "E1_mixed"):
            run_command(args, "validation_" + condition, [sys.executable, "-u", "-B", str(ENTRY),
                "evaluate-gain", "--root", str(args.root), "--device", "cuda", "--checkpoint", str(gain),
                "--split", "validation", "--condition", condition, "--samples", "32", "--pairs", "8",
                "--compare-allocations", "--output", str(args.output / f"validation_gain_{condition}.json")])
        # The requested unattended run records these diagnostics, then trains the
        # joint model. Geometry improvement is assessed from the saved reports.
        joint = train_stage(args, "joint", 30, gain)
        print(f"PIPELINE COMPLETE; final best checkpoint: {joint}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"PIPELINE FAILED: {error}", file=sys.stderr, flush=True)
        raise
