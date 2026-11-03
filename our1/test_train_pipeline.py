import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("ours1_pipeline", Path(__file__).with_name("train_pipeline.py"))
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def last_checkpoint(stage, epoch, epochs):
    return dict(protocol="V3", network_version=pipeline.VERSION, partial_debug_run=False,
                stage=stage, epoch=epoch, args=dict(epochs=epochs),
                configuration=dict(allocation="gdb", relation=stage == "joint"))


class PipelineTests(unittest.TestCase):
    def test_child_output_is_logged_and_nonzero_exit_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder))
            with self.assertRaises(subprocess.CalledProcessError):
                pipeline.run_command(args, "child", [sys.executable, "-u", "-c", "print('progress', flush=True); raise SystemExit(2)"])
            self.assertIn("progress", (Path(folder) / "child.console.log").read_text(encoding="utf-8"))

    def test_new_interrupted_old_completed_and_selected_stages(self):
        self.assertEqual(pipeline.stage_action(None, None, "rgr", 20, "data"), "start")
        last = last_checkpoint("rgr", 9, 20)
        self.assertEqual(pipeline.stage_action(last, None, "rgr", 20, "data"), "resume")
        last["epoch"] = 19
        self.assertEqual(pipeline.stage_action(last, None, "rgr", 20, "data"), "select")
        best = dict(last, epoch=9, selection_completed_epochs=20,
                    best_validation=dict(protocol=dict(full_validation=True, data_manifest_sha256="data")))
        self.assertEqual(pipeline.stage_action(last, best, "rgr", 20, "data"), "done")
        best["selection_completed_epochs"] = 10
        self.assertEqual(pipeline.stage_action(last, best, "rgr", 20, "data"), "resume")
        best["best_validation"]["protocol"]["full_validation"] = False
        with self.assertRaises(ValueError):
            pipeline.stage_action(last, best, "rgr", 20, "data")

    def test_wrong_model_debug_and_wrong_training_length_are_rejected(self):
        for change in (dict(stage="gain"), dict(epoch=20), dict(partial_debug_run=True),
                       dict(args=dict(epochs=60)), dict(network_version="old")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                pipeline.stage_action({**last_checkpoint("rgr", 19, 20), **change}, None, "rgr", 20, "data")

    def test_train_failure_does_not_advance_and_retains_input_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent = root / "common.pt"
            parent.write_bytes(b"common")
            args = SimpleNamespace(output=root, root=root, data_hash="data", source_hash="source", training_profile={})
            with patch.object(pipeline, "wait_for_workers"), patch.object(pipeline, "run_command", side_effect=RuntimeError("failed")) as run:
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    pipeline.train_stage(args, "rgr", 20, parent)
                self.assertEqual(run.call_count, 1)
                self.assertTrue((root / "rgr/pipeline_inputs.json").exists())

    def test_successful_stage_runs_command_and_requires_completed_best(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent = root / "parent.pt"
            parent.write_bytes(b"parent")
            args = SimpleNamespace(output=root, root=root, data_hash="data", source_hash="source", training_profile={})
            last = last_checkpoint("joint", 29, 30)
            best = dict(last, selection_completed_epochs=30,
                        best_validation=dict(protocol=dict(full_validation=True, data_manifest_sha256="data")))
            with patch.object(pipeline, "wait_for_workers"), patch.object(pipeline, "load", side_effect=[last, best]), patch.object(pipeline, "run_command") as run:
                self.assertEqual(pipeline.train_stage(args, "joint", 30, parent), root / "joint/best.pt")
                command = run.call_args.args[2]
                self.assertIn("gdb_flow", command)
                self.assertIn("--checkpoint", command)
