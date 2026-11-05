"""OUNet evaluation uses the common object selection and its trained octree scale."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_subset_eval as runner


def ounet_checkpoint():
    return {"protocol": "V3", "epoch": 59,
            "args": {"model_name": "ounet", "epochs": 60, "training_mode": "mixed", "max_batches": 0,
                     "ounet_schedule_epochs": 60, "ounet_poly_power": .9, **runner.OUNET_INFERENCE},
            "scheduler": {"schedule_type": "ounet-poly-v1", "schedule_epochs": 60, "power": .9, "last_epoch": 60},
            "result": {"partial_debug_run": False}}


def data_arguments(root):
    (root / "metadata").mkdir()
    data = {"groups": {
        "train": [{"rows": 64, "object_id": "train/a"}],
        "pu1k_object": [{"rows": 1, "object_id": f"pu1k/{i:03d}"} for i in range(100)],
        "modelnet40_patch": [{"rows": 1, "object_id": f"modelnet40/{i:03d}"} for i in range(120)]}}
    for name, value in (("base_manifest.json", data), ("generation_config.json", {}), ("dataset_splits.json", {})):
        (root / "metadata" / name).write_text(json.dumps(value), encoding="utf-8")
    return SimpleNamespace(repo=ROOT, root=root, output=root / "evaluation", split="all",
                           objects=80, patches_per_object=None, seed=21, models=["ounet"],
                           conditions=runner.CONDITIONS, calibration_samples=8, schedule=None,
                           whole_data_root=None, device="cuda", p2f=False,
                           **{f"{name}_checkpoint": ROOT / path for name, path in runner.CHECKPOINTS.items()})


class OUNetSubsetTests(unittest.TestCase):
    def test_completed_fixed_budget_checkpoint_is_accepted_without_mutation(self):
        saved = ounet_checkpoint()
        before = copy.deepcopy(saved)
        runner.check_checkpoint(saved, "ounet")
        self.assertEqual(saved, before)
        with self.assertRaisesRegex(ValueError, "60-epoch"):
            runner.check_checkpoint(dict(saved, epoch=58), "ounet")
        with self.assertRaises(ValueError):
            runner.check_checkpoint(saved, "spu")

    def test_debug_wrong_scale_or_incomplete_poly_checkpoint_is_rejected(self):
        changes = (("model_name", "pufm++"), ("epochs", 30), ("training_mode", "clean"),
                   ("max_batches", 20), ("octree_depth", 9), ("octree_scale", .5),
                   ("octree_beam", 8), ("ounet_schedule_epochs", 80), ("ounet_poly_power", 1.))
        for key, value in changes:
            saved = ounet_checkpoint()
            saved["args"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                runner.check_checkpoint(saved, "ounet")
        for changes in ({"result": {}}, {"result": {"partial_debug_run": True}}, {"scheduler": None},
                        {"scheduler": dict(ounet_checkpoint()["scheduler"], last_epoch=59)},
                        {"scheduler": dict(ounet_checkpoint()["scheduler"], power=1.)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runner.check_checkpoint(ounet_checkpoint() | changes, "ounet")

    def test_same_eighty_objects_conditions_and_sample_indices_as_other_models(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            args = data_arguments(Path(directory))
            actual = runner.make_suite(args)
            self.assertEqual(actual["splits"], ["pu1k_object", "modelnet40_object"])
            self.assertEqual(actual["total_predictions"], 1120)
            self.assertEqual(actual["calibration_runs"], 0)
            for name in ("pufm", "pufmpp", "ours", "spu", "ours1"):
                args.models = [name]
                reference = runner.make_suite(args)
                for first, second in zip(actual["plans"], reference["plans"]):
                    self.assertEqual(first["selection"], second["selection"])
                    self.assertEqual(first["selection_sha256"], second["selection_sha256"])
                    self.assertEqual(first["conditions"], second["conditions"])
                    self.assertEqual(first["selection"]["samples_per_condition"], 80)
                    self.assertEqual(first["inference_runtime"]["ounet"], runner.OUNET_INFERENCE)

    def test_loader_preserves_fixed_budget_scale_and_passes_only_input_to_predict(self):
        observed = object()
        calls = []

        class Network:
            def eval(self):
                calls.append("eval")

            def predict(self, points, count):
                calls.append((points, count))
                return points, count

        network = Network()

        def load_adapter(folder, args):
            self.assertEqual(folder, ROOT / "compare-change/upsample-clean-master")
            self.assertEqual(args.command, "predict")
            self.assertEqual(args.model_name, "ounet")
            for key, value in runner.OUNET_INFERENCE.items():
                self.assertEqual(getattr(args, key), value)
            return network

        def load_weights(adapter, path, args):
            self.assertIs(adapter, network)
            self.assertEqual(path, "trained_ounet_last.pt")
            return ounet_checkpoint()

        common = SimpleNamespace(set_seed=lambda seed: calls.append(seed),
                                 load_adapter=load_adapter, load_weights=load_weights)
        plan = {"comparison_root": str(ROOT / "compare-change"), "device": "cpu", "seed": 21,
                "checkpoints": {"ounet": "trained_ounet_last.pt"},
                "inference_runtime": {"ounet": dict(runner.OUNET_INFERENCE)}}
        with patch.dict(sys.modules, {"torch": SimpleNamespace()}), \
             patch.object(runner, "import_file", return_value=common), \
             patch.object(sys, "path", sys.path.copy()), redirect_stdout(io.StringIO()):
            loaded, predict, loaded_common = runner.load_network(plan, "ounet")
            self.assertIs(loaded, network)
            self.assertIs(loaded_common, common)
            self.assertEqual(predict(observed, 8192), (observed, 8192))
            self.assertIn("eval", calls)
            plan["inference_runtime"]["ounet"]["octree_scale"] = .5
            with self.assertRaisesRegex(ValueError, "inference settings"):
                runner.load_network(plan, "ounet")

    def test_cli_accepts_ounet_and_resolves_default_checkpoint_and_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            args = data_arguments(Path(directory))
            completed = subprocess.run([sys.executable, "-B", str(ROOT / "run_subset_eval.py"),
                                        "--repo", str(ROOT), "--root", str(args.root), "--models", "ounet",
                                        "--calibration-samples", "8", "--phase", "plan"],
                                       capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("TOTAL SUBSET PREDICTIONS: 1120; PUFM++ calibration runs: 0", completed.stdout)
            self.assertIn("whole_object_eval80_ounet", completed.stdout)
            self.assertIn('"octree_scale": 0.25', completed.stdout)
            self.assertIn("ounet_b8_poly60_scale025_seed21", completed.stdout)


if __name__ == "__main__":
    unittest.main()
