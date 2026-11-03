import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from adaptive_resampling.v3_core import VALIDATION_CONDITIONS
from validation_selection import ValidationSelection, aggregate, atomic_save, evaluation_state


class ValidationTests(unittest.TestCase):
    def test_family_and_object_weights_do_not_count_patches_as_objects(self):
        rows = [(c, "one", float(c[0] == "D")) for c in VALIDATION_CONDITIONS]
        rows.extend([("A0_normal", "two", 2.)] * 4)
        score, conditions, _ = aggregate(rows)
        self.assertEqual(conditions["A0_normal"], 1.)
        self.assertAlmostEqual(score, .4)

    def test_evaluation_preserves_rng_modes_and_configuration_even_on_failure(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Dropout())
        model.config = object()
        config = model.config
        model.train()
        model[0].eval()
        modes = [m.training for m in model.modules()]
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        expected = random.random(), np.random.rand(), torch.rand(1)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        with self.assertRaises(RuntimeError):
            with evaluation_state(model):
                random.random(), np.random.rand(), torch.rand(1)
                model.config = None
                raise RuntimeError("interrupted")
        actual = random.random(), np.random.rand(), torch.rand(1)
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertIs(model.config, config)

    def test_resume_preserves_an_earlier_best_and_marks_completed_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata").mkdir()
            (root / "metadata/base_manifest.json").write_text(json.dumps(
                dict(groups=dict(validation=[dict(object_id="v1", rows=4)]))))
            args = SimpleNamespace(root=root, output=root / "run", seed=21, val_objects=0,
                                   val_every=5, stage="joint", epochs=10, resume=None)
            selector = ValidationSelection(args)
            self.assertFalse(selector.due(0))
            self.assertTrue(selector.due(4))
            self.assertTrue(selector.due(9))
            report = dict(epoch=4, value=.01, protocol=selector.protocol, samples=28)
            best = dict(epoch=4, stage="joint", best_validation=report,
                        validation=report, args=dict(epochs=10), model={"weight": torch.tensor([5.])})
            atomic_save(best, args.output / "best.pt")
            last = {**best, "epoch": 9, "model": {"weight": torch.tensor([10.])}}
            args.resume = args.output / "last.pt"
            resumed = ValidationSelection(args, last)
            self.assertEqual(resumed.best["value"], .01)
            resumed.complete()
            selected = torch.load(args.output / "best.pt", weights_only=False)
            self.assertEqual(selected["epoch"], 4)
            self.assertEqual(selected["selection_completed_epochs"], 10)
            self.assertEqual(selected["model"]["weight"].item(), 5.)
            changed = copy.copy(args)
            changed.val_objects = 1
            # One object is the whole population; choose a distinct protocol change.
            changed.stage = "rgr"
            with self.assertRaisesRegex(ValueError, "protocol"):
                ValidationSelection(changed, last)
