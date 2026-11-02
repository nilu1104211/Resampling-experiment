"""Ours evaluation must preserve trained modules and the existing whole-object task."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_subset_eval as runner


def ours_checkpoint():
    return {"protocol": "V3", "epoch": 29, "stage": "joint", "partial_debug_run": False,
            "configuration": {"use_rgr": True, "allocation": "gain", "condition": False},
            "trained_stages": ["baseline", "rgr", "gain", "joint"]}


class OursSubsetTests(unittest.TestCase):
    def test_completed_joint_is_accepted_without_relabeling_as_baseline(self):
        saved = ours_checkpoint()
        original = copy.deepcopy(saved)
        runner.check_checkpoint(saved, "ours")
        self.assertEqual(saved, original)
        with self.assertRaisesRegex(ValueError, "plain baseline|non-debug baseline"):
            runner.check_checkpoint(saved, "pufm")

    def test_intermediate_debug_or_incomplete_ours_weights_are_rejected(self):
        for change in ({"stage": "rgr"}, {"stage": "gain"}, {"partial_debug_run": True},
                       {"epoch": 28}, {"trained_stages": ["baseline", "joint"]},
                       {"configuration": {"use_rgr": False, "allocation": "midpoint"}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.check_checkpoint(dict(ours_checkpoint(), **change), "ours")

    def test_baseline_and_pufmpp_checks_remain_distinct(self):
        baseline = dict(ours_checkpoint(), stage="baseline", configuration={
            "use_rgr": False, "condition": False, "allocation": "midpoint"})
        runner.check_checkpoint(baseline, "pufm")
        with self.assertRaises(ValueError):
            runner.check_checkpoint(baseline, "ours")
        enhanced = {"protocol": "V3", "epoch": 29, "args": {"stage": "s2"},
                    "result": {"partial_debug_run": False}}
        runner.check_checkpoint(enhanced, "pufmpp")
        with self.assertRaises(ValueError):
            runner.check_checkpoint(enhanced, "ours")

    def test_ours_uses_the_same_80_objects_seven_conditions_and_whole_clouds(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temporary:
            root = Path(temporary)
            (root / "metadata").mkdir()
            data = {"groups": {
                "train": [{"rows": 64, "object_id": "train/a"}],
                "pu1k_object": [{"rows": 1, "object_id": f"pu1k/{i:03d}"} for i in range(100)],
                "modelnet40_patch": [{"rows": 1, "object_id": f"modelnet40/{i:03d}"} for i in range(120)]}}
            for name, value in (("base_manifest.json", data), ("generation_config.json", {}),
                                ("dataset_splits.json", {})):
                (root / "metadata" / name).write_text(json.dumps(value), encoding="utf-8")
            args = SimpleNamespace(repo=ROOT, root=root, output=root / "evaluation", split="all",
                                   objects=80, patches_per_object=None, seed=21, models=["ours"],
                                   conditions=runner.CONDITIONS, calibration_samples=8, schedule=None,
                                   whole_data_root=None, device="cuda", p2f=False,
                                   **{f"{name}_checkpoint": ROOT / value for name, value in runner.CHECKPOINTS.items()})
            ours = runner.make_suite(args)
            args.models = ["pufm"]
            baseline = runner.make_suite(args)
            self.assertEqual(ours["total_predictions"], 1120)
            self.assertEqual(ours["calibration_runs"], 0)
            self.assertEqual(ours["splits"], ["pu1k_object", "modelnet40_object"])
            for actual, reference in zip(ours["plans"], baseline["plans"]):
                self.assertEqual(actual["selection"], reference["selection"])
                self.assertEqual(actual["selection_sha256"], reference["selection_sha256"])
                self.assertEqual(actual["conditions"], reference["conditions"])
                self.assertEqual(actual["selection"]["samples_per_condition"], 80)
                self.assertEqual(list(actual["checkpoints"]), ["ours"])

    def test_loader_inherits_saved_configuration_and_keeps_input_only_inference(self):
        observed = object()
        calls = []

        class Network:
            config = SimpleNamespace(use_rgr=True, allocation="gain", condition=False, flow_steps=5)

            def eval(self):
                calls.append("eval")

            def prepare(self, points, count):
                self_input = (points, count)
                calls.append(self_input)
                return self_input

            def integrate(self, state):
                return state

        network = Network()
        entry = SimpleNamespace(seed_all=lambda seed: calls.append(seed))

        def create_model(args):
            self.assertIsNone(args.preset)  # Do not replace the trained configuration with baseline.
            self.assertIsNone(args.allocation)
            self.assertIsNone(args.flow_steps)
            self.assertEqual(args.command, "predict")
            self.assertEqual(args.stage, "joint")
            self.assertEqual(args.checkpoint, "trained_full.pt")
            self.assertFalse(args.no_recovery or args.no_virtual or args.allow_debug)
            return network, ours_checkpoint()

        entry.create_model = create_model
        plan = {"checkpoints": {"ours": "trained_full.pt"}, "pufm_root": str(ROOT / "our/PUFM-main"),
                "device": "cpu", "seed": 21}
        # The loader imports torch but never calls it; this test needs only the stdlib.
        with patch.dict(sys.modules, {"torch": SimpleNamespace()}), \
             patch.object(runner, "import_file", return_value=entry), \
             patch.object(sys, "path", sys.path.copy()), redirect_stdout(io.StringIO()):
            loaded, predict, common = runner.load_network(plan, "ours")
        self.assertIs(loaded, network)
        self.assertIsNone(common)
        self.assertEqual(predict(observed, 8192), (observed, 8192))
        self.assertIn("eval", calls)


if __name__ == "__main__":
    unittest.main()
