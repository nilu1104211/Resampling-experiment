"""Verify OUNet epoch timing and uninterrupted versus resumed training."""
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "compare-change"))
import v3_common as common
from adaptive_resampling.v3_scheduler import OUNetPolyLR

# Import the production adapter without retaining its generic `model` package in
# the test process; other comparison adapters can use that name as well.
folder = ROOT / "compare-change/upsample-clean-master"
previous_models = {name: module for name, module in sys.modules.items()
                   if name == "model" or name.startswith("model.")}
with patch.object(sys, "path", [str(folder), *sys.path]):
    for name in list(sys.modules):
        if name == "model" or name.startswith("model."):
            del sys.modules[name]
    try:
        spec = importlib.util.spec_from_file_location("ounet_scheduler_test_adapter", folder / "v3_adapter.py")
        ounet = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ounet)
    finally:
        for name in list(sys.modules):
            if name == "model" or name.startswith("model."):
                del sys.modules[name]
        sys.modules.update(previous_models)


class TinyDataset:
    training_profile = {"selection_sha256": "test", "source_config_digest": "test",
                        "samples": 4, "source_samples": 4, "training_mode": "mixed"}

    def __len__(self):
        return 4

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, key):
        epoch, index = key
        return {"input_xyz": torch.full((256, 3), (index + epoch + 1) / 10),
                "gt_xyz": torch.full((1024, 3), (index + 1) / 20)}

    def close(self):
        pass


class TinyOUNet(ounet.Adapter):
    """Use the real adapter's scheduling hooks with a cheap stochastic loss."""

    def __init__(self, args):
        nn.Module.__init__(self)
        self.model = nn.Linear(3, 3)

    def training_loss(self, points, gt):
        noise = torch.rand(()) + random.random() + np.random.rand()
        return (self.model(points.mean(1)) - gt.mean(1) + noise * .01).square().mean()


def arguments(output, **changes):
    values = dict(model_name="ounet", device="cpu", batch_size=2, workers=0,
                  lr=5e-4, weight_decay=.05, fused_optimizer=False, no_fused_optimizer=False,
                  epochs=4, ounet_schedule_epochs=4, ounet_poly_power=.9,
                  octree_depth=8, octree_scale=.5, training_mode="mixed", seed=21,
                  max_batches=0, validation_samples=0, resume=None, checkpoint=None,
                  log_every=20, output=str(output))
    return SimpleNamespace(**(values | changes))


def run_training(args, seed=21):
    common.set_seed(seed)
    adapter = TinyOUNet(args)
    with patch.object(common, "dataset", return_value=TinyDataset()), redirect_stdout(io.StringIO()):
        common.train(adapter, args)
    return torch.load(Path(args.output) / "last.pt", map_location="cpu", weights_only=False)


class OUNetSchedulerTests(unittest.TestCase):
    def test_domain_errors_identify_role_scale_and_batch_slot_without_clipping(self):
        args = arguments("unused")
        adapter = TinyOUNet(args)
        adapter.args = args
        points = torch.tensor([[[0., 0., 0.], [2.0793188, 0., 0.]]])
        original = points.clone()
        with self.assertRaisesRegex(ValueError, r"GT coordinate.*octree_scale=0.5.*batch_slots=\[0\]"):
            adapter.build_octree(points, features=False)
        args.octree_scale = .25
        tree = adapter.build_octree(points, features=False)
        self.assertEqual(tree.depth, args.octree_depth)
        torch.testing.assert_close(points, original, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            adapter.build_octree(points * float("nan"), features=True)

    def test_source_formula_epoch_boundaries_and_zero_after_horizon(self):
        parameter = nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=5e-4)
        scheduler = OUNetPolyLR(optimizer, 4)
        rates = []
        for _ in range(5):
            rates.append(optimizer.param_groups[0]["lr"])
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scheduler.step()
        for epoch, rate in enumerate(rates):
            self.assertAlmostEqual(rate, 5e-4 * (1 - epoch / 4) ** .9, places=15)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0)
        for horizon, power in ((0, .9), (4, 0), (4, float("nan")), (4, float("inf"))):
            with self.subTest(horizon=horizon, power=power), self.assertRaises(ValueError):
                OUNetPolyLR(optimizer, horizon, power)

    def test_staged_resume_matches_all_weights_optimizer_rng_and_logged_rates(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            full = run_training(arguments(root / "full"))
            first = run_training(arguments(root / "staged", epochs=2))
            self.assertEqual(first["scheduler"]["last_epoch"], 2)
            self.assertAlmostEqual(first["optimizer"]["param_groups"][0]["lr"], 5e-4 * .5 ** .9)
            resumed = run_training(arguments(root / "staged", resume=str(root / "staged/last.pt")), seed=999)
            for name, tensor in full["model"].items():
                torch.testing.assert_close(tensor, resumed["model"][name], rtol=0, atol=0)
            self.assertEqual(full["optimizer"]["param_groups"], resumed["optimizer"]["param_groups"])
            for parameter, state in full["optimizer"]["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, resumed["optimizer"]["state"][parameter][key], rtol=0, atol=0)
            self.assertEqual(full["scheduler"], resumed["scheduler"])
            self.assertTrue(torch.equal(full["rng"]["torch"], resumed["rng"]["torch"]))
            logs = [[json.loads(line) for line in (root / run / "train.jsonl").read_text().splitlines()]
                    for run in ("full", "staged")]
            self.assertEqual(logs[0], logs[1])
            self.assertEqual([row["epoch"] for row in logs[0]], list(range(4)))
            self.assertEqual(logs[0][-1]["next_lr"], 0)
            self.assertGreater(logs[0][-1]["lr"], 0)

    def test_resume_rejects_changed_recipe_missing_state_or_bad_progress(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            saved = run_training(arguments(root / "first", epochs=1))
            checkpoint = root / "resume.pt"
            torch.save(saved, checkpoint)
            for key, value in (("ounet_schedule_epochs", 5), ("ounet_poly_power", 1.),
                               ("lr", 1e-3), ("weight_decay", 0.), ("batch_size", 1)):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    run_training(arguments(root / "next", resume=str(checkpoint), **{key: value}))
            for change, message in (({"scheduler": None}, "saved ounet scheduler"),
                                    ({"scheduler": dict(saved["scheduler"], last_epoch=0)}, "progress"),
                                    ({"scheduler": dict(saved["scheduler"], power=2.)}, "power"),
                                    ({"scheduler": dict(saved["scheduler"], schedule_epochs=9)}, "schedule_epochs")):
                torch.save(saved | change, checkpoint)
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                    run_training(arguments(root / "next", resume=str(checkpoint)))

    def test_debug_checkpoint_cannot_resume_as_formal_training(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            saved = run_training(arguments(root / "debug", epochs=1, max_batches=1))
            self.assertTrue(saved["result"]["partial_debug_run"])
            with self.assertRaisesRegex(ValueError, "partial debug"):
                run_training(arguments(root / "formal", resume=str(root / "debug/last.pt")))
            run_training(arguments(root / "debug", epochs=2, max_batches=1, resume=str(root / "debug/last.pt")))

    def test_cli_resolves_horizon_and_rejects_invalid_schedule_before_loading_data(self):
        for options, expected in (([], 60), (["--epochs", "2", "--ounet-schedule-epochs", "80"], 80)):
            with patch.object(sys, "argv", ["v3.py", "train", *options]), \
                 patch.object(common, "load_adapter", return_value=object()), patch.object(common, "train") as train:
                common.main(folder, "ounet")
                self.assertEqual(train.call_args.args[1].ounet_schedule_epochs, expected)
        for options in (["--epochs", "5", "--ounet-schedule-epochs", "4"],
                        ["--ounet-schedule-epochs", "0"], ["--ounet-poly-power", "nan"],
                        ["--ounet-poly-power", "0"]):
            with self.subTest(options=options), patch.object(sys, "argv", ["v3.py", "train", *options]), \
                 redirect_stdout(io.StringIO()), patch("sys.stderr", new_callable=io.StringIO), \
                 patch.object(common, "load_adapter") as load, self.assertRaises(SystemExit):
                common.main(folder, "ounet")
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
