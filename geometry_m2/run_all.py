"""Run common preparation, then D1/D2/D3 training sequentially with resume."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--pu1k-h5", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"server_runs"/"geometry_m2_seed21")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preadapt-epochs", type=int, default=10)
    parser.add_argument("--gain-epochs", type=int, default=20)
    parser.add_argument("--adapt-epochs", type=int, default=10)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--teacher-samples", type=int, default=4096)
    parser.add_argument("--teacher-pairs", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=5)
    parser.add_argument("--skip-screen", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    args = parser.parse_args()
    args.root, args.checkpoint, args.output = args.root.resolve(), args.checkpoint.resolve(), args.output.resolve()
    if args.pu1k_h5 is None:
        matches = sorted(args.root.rglob("pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5"))
        if len(matches) != 1:
            parser.error("Specify --pu1k-h5 explicitly when the dataset root has zero or multiple matching HDF5 files")
        args.pu1k_h5 = matches[0]
    if not args.checkpoint.is_file() or not args.pu1k_h5.is_file():
        parser.error("Checkpoint or HDF5 file missing")
    args.output.mkdir(parents=True, exist_ok=True)
    common = ["--root", str(args.root), "--pu1k-h5", str(args.pu1k_h5.resolve()),
              "--device", args.device, "--seed", str(args.seed), "--batch-size", str(args.batch_size),
              "--workers", str(args.workers), "--freeze-epochs", str(args.freeze_epochs),
              "--flow-steps", str(args.flow_steps),
              "--max-batches", str(args.max_batches), "--validation-limit", str(args.validation_limit)]

    def run(mode, command, checkpoint, output, *extra):
        entry = ROOT/"geometry_m2"/mode/"PUFM-main"/"train.py"
        cmd = [sys.executable, "-u", "-B", str(entry), command, *common,
               "--checkpoint", str(checkpoint), "--output", str(output), *map(str, extra)]
        if command == "train" and (output/"last.pt").exists():
            cmd += ["--resume", str(output/"last.pt")]
        print("\n" + subprocess.list2cmdline(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)

    if not args.skip_screen:
        for mode in ("D1", "D2", "D3"):
            run(mode, "screen", args.checkpoint, args.output/"screen"/mode)
    shared = args.output/"common_preadapt"
    run("D1", "train", args.checkpoint, shared, "--stage", "preadapt", "--epochs", args.preadapt_epochs)
    teacher = args.output/"teacher"
    run("D1", "teacher", shared/"best.pt", teacher, "--teacher-samples", args.teacher_samples,
        "--teacher-pairs", args.teacher_pairs, "--teacher-batch-size", args.batch_size)
    for mode in ("D1", "D2", "D3"):
        gain, adapt = args.output/mode/"gain", args.output/mode/"adapt"
        run(mode, "train", shared/"best.pt", gain, "--stage", "gain", "--epochs", args.gain_epochs,
            "--teacher", teacher, "--lr", 1e-3)
        run(mode, "train", shared/"best.pt", adapt, "--stage", "adapt", "--epochs", args.adapt_epochs,
            "--allocator", gain/"best.pt")
        run(mode, "drift", adapt/"best.pt", adapt, "--teacher", teacher)
    print(f"Completed D1/D2/D3. Best adapted weights are under {args.output}/D*/adapt/best.pt", flush=True)


if __name__ == "__main__":
    main()
