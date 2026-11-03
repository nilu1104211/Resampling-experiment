import copy
import unittest

import run_subset_eval as runner


def checkpoint():
    selected = dict(epoch=9, value=.02, samples=2800, protocol=dict(
        split="validation", full_validation=True, conditions=runner.CONDITIONS,
        metric="canonical_cd_l1_object_condition_family_mean", expected_samples=2800))
    return dict(protocol="V3", epoch=9, args=dict(epochs=30), stage="joint",
                network_version="ours1-rgdb-relflow-v1", partial_debug_run=False,
                checkpoint_role="validation_best", best_validation=selected,
                validation=copy.deepcopy(selected), selection_completed_epochs=30,
                configuration=dict(allocation="gdb", relation=True),
                trained_stages=["baseline", "rgr", "gain", "joint"])


class BestCheckpointTests(unittest.TestCase):
    def test_selected_earlier_epoch_of_completed_stage_is_accepted(self):
        runner.check_checkpoint(checkpoint(), "ours1")

    def test_unfinished_partial_test_selected_and_unmarked_checkpoints_are_rejected(self):
        for change in (dict(selection_completed_epochs=10), dict(partial_debug_run=True),
                       dict(checkpoint_role="training"), dict(validation=None)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.check_checkpoint({**checkpoint(), **change}, "ours1")
        for change in (dict(full_validation=False), dict(split="pu1k_object"), dict(conditions=["A0_normal"])):
            candidate = checkpoint()
            candidate["best_validation"]["protocol"].update(change)
            candidate["validation"] = copy.deepcopy(candidate["best_validation"])
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.check_checkpoint(candidate, "ours1")
