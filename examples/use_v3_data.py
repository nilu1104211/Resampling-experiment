"""Check V3 data or inspect a real PyTorch DataLoader batch.

python examples/use_v3_data.py --numpy-only
python examples/use_v3_data.py --workers 4
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_resampling.v3_dataset import DEFAULT_ROOT, AdaptiveResamplingDataset, EpochSampler, inverse_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--numpy-only", action="store_true")
    args = parser.parse_args()
    train = AdaptiveResamplingDataset(args.root, "train", training_mode="mixed", include_supervision=True)
    sampler = EpochSampler(train)
    print("Training clean bases:", len(train))
    if args.numpy_only:
        for epoch in range(5):
            sampler.set_epoch(epoch)
            sample = train[(epoch, 0)]
            print(epoch, sample["condition"], sample["input_xyz"].shape, sample["gt_xyz"].shape,
                  "auxiliary targets:", sorted(sample["supervision"]))
        test = AdaptiveResamplingDataset(args.root, "pu1k_patch", condition="C2_hole_10",
                                         include_supervision=True, include_reference_metadata=True)
        sample = test[0]
        canonical_target = inverse_prediction(sample["gt_xyz"], sample)
        print("Fixed PU1K hole test:", len(test), canonical_target.shape)
        print("Forward network input_xyz only; gt_xyz and supervision are training/evaluation targets.")
        test.close()
    else:
        try:
            from torch.utils.data import DataLoader
        except ImportError as error:
            raise SystemExit("Use your network's PyTorch environment, or add --numpy-only for data checks.") from error
        loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                            persistent_workers=args.workers > 0, pin_memory=True)
        for epoch in range(2):
            sampler.set_epoch(epoch)
            batch = next(iter(loader))
            x, y = batch["input_xyz"], batch["gt_xyz"]
            print("Epoch", epoch, "input", tuple(x.shape), "target", tuple(y.shape))
            # For your network: prediction = model(x.to(device)).
            # Many existing baselines instead expect x.transpose(1, 2).
            # Keep auxiliary labels out of the forward call.
    train.close()


if __name__ == "__main__":
    main()
