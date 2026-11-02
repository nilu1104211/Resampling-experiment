"""SPU testing must load completed shared-input weights and preserve native recovery."""
import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_subset_eval as runner


def spu_checkpoint():
    return {"protocol": "V3", "epoch": 59,
            "args": {"model_name": "spu-imr", "training_mode": "mixed", "max_batches": 0,
                     **runner.SPU_INFERENCE},
            "result": {"partial_debug_run": False}}


class SPUSubsetTests(unittest.TestCase):
    def test_completed_spu_uses_its_own_60_epoch_budget(self):
        saved = spu_checkpoint()
        original = copy.deepcopy(saved)
        runner.check_checkpoint(saved, "spu")
        self.assertEqual(saved, original)
        with self.assertRaisesRegex(ValueError, "60-epoch"):
            runner.check_checkpoint(dict(saved, epoch=29), "spu")
        with self.assertRaisesRegex(ValueError, "30-epoch"):
            runner.check_checkpoint(saved, "pufm")

    def test_debug_legacy_native_or_changed_recipe_is_rejected(self):
        for key, value in (("model_name", "pufm++"), ("spu_profile", "native"),
                           ("spu_profile", "legacy-patch256"), ("spu_groups", 256),
                           ("spu_transport", "hungarian"), ("spu_passes", 1),
                           ("training_mode", "clean"), ("max_batches", 1)):
            saved = spu_checkpoint()
            saved["args"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                runner.check_checkpoint(saved, "spu")
        for result in ({}, {"partial_debug_run": True}):
            with self.subTest(result=result), self.assertRaises(ValueError):
                runner.check_checkpoint(dict(spu_checkpoint(), result=result), "spu")

    def test_loader_uses_spu_adapter_and_input_only_predict_with_eight_recoveries(self):
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
            self.assertEqual(folder, ROOT / "compare-change/spu-imr-main")
            self.assertEqual(args.command, "predict")
            self.assertEqual(args.model_name, "spu-imr")
            for key, value in runner.SPU_INFERENCE.items():
                self.assertEqual(getattr(args, key), value)
            return network

        def load_weights(adapter, path, args):
            self.assertIs(adapter, network)
            self.assertEqual(path, "trained_spu_last.pt")
            return spu_checkpoint()

        common = SimpleNamespace(set_seed=lambda seed: calls.append(seed),
                                 load_adapter=load_adapter, load_weights=load_weights)
        plan = {"comparison_root": str(ROOT / "compare-change"), "device": "cpu", "seed": 21,
                "checkpoints": {"spu": "trained_spu_last.pt"},
                "inference_runtime": {"spu": dict(runner.SPU_INFERENCE)}}
        with patch.dict(sys.modules, {"torch": SimpleNamespace()}), \
             patch.object(runner, "import_file", return_value=common), \
             patch.object(sys, "path", sys.path.copy()), redirect_stdout(io.StringIO()):
            loaded, predict, loaded_common = runner.load_network(plan, "spu")
            self.assertIs(loaded, network)
            self.assertIs(loaded_common, common)
            self.assertEqual(predict(observed, 8192), (observed, 8192))
            self.assertIn("eval", calls)
            plan["inference_runtime"]["spu"]["spu_passes"] = 1
            with self.assertRaisesRegex(ValueError, "inference settings"):
                runner.load_network(plan, "spu")


if __name__ == "__main__":
    unittest.main()
