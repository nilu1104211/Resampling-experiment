"""Frozen data remains valid across the reviewed opt-in OUNet preflight change."""
import copy
from pathlib import Path
import tempfile
import unittest

import run_subset_eval as runner


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SOURCE = "02135eeb32c35a8b8b5a560fac693d7e9e3713d81ee505deb83a7752973ce66c"


class SupplementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "object.npz").write_bytes(b"immutable frozen object")
        self.package = dict(schema="frozen-v3-whole-objects-v1", input_points=2048, target_points=8192,
                            source_manifest_sha256="manifest", source_splits_sha256="splits",
                            protocol_source_sha256=FROZEN_SOURCE, conditions=runner.CONDITIONS,
                            records=[dict(object_id="test/object", file="object.npz",
                                          sha256=runner.file_hash(self.root / "object.npz"))])
        self.plan = dict(split="modelnet40_object", repo=str(ROOT), whole_data_root=str(self.root),
                         metadata_sha256={"base_manifest.json": "manifest", "dataset_splits.json": "splits"},
                         conditions=runner.CONDITIONS, selection=dict(objects=[dict(object_id="test/object")]))
        self.write_manifest()

    def write_manifest(self):
        runner.write_json(self.root / "manifest.json", self.package)
        self.plan["supplement_manifest_sha256"] = runner.file_hash(self.root / "manifest.json")

    def test_reviewed_default_path_is_accepted_and_recorded_without_modifying_data(self):
        before = {path.name: path.read_bytes() for path in self.root.iterdir()}
        runner.validate_supplement(self.plan)
        self.assertIn("coordinates_only=False", self.plan["supplement_protocol"]["compatibility"])
        self.assertEqual(self.plan["supplement_protocol"]["package_source_sha256"], FROZEN_SOURCE)
        self.assertEqual(self.plan["supplement_files_sha256"], {"object.npz": self.package["records"][0]["sha256"]})
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.root.iterdir()})

    def test_exact_source_match_is_still_accepted(self):
        self.package["protocol_source_sha256"] = runner.signature({
            name: (ROOT / "adaptive_resampling" / name).read_text(encoding="utf-8").strip()
            for name in ("core.py", "v3_core.py")})
        self.write_manifest()
        runner.validate_supplement(self.plan)
        self.assertEqual(self.plan["supplement_protocol"]["compatibility"], "exact_source_match")

    def test_unknown_package_protocol_is_rejected(self):
        self.package["protocol_source_sha256"] = "0" * 64
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "sampling/corruption protocol differs"):
            runner.validate_supplement(self.plan)

    def test_unreviewed_source_change_is_rejected(self):
        repo = self.root / "changed_repo"
        folder = repo / "adaptive_resampling"
        folder.mkdir(parents=True)
        for name in ("core.py", "v3_core.py"):
            source = (ROOT / "adaptive_resampling" / name).read_text(encoding="utf-8")
            if name == "v3_core.py":
                self.assertIn("SEED = 20261021", source)
                source = source.replace("SEED = 20261021", "SEED = 20261022")
            (folder / name).write_text(source, encoding="utf-8")
        self.plan["repo"] = str(repo)
        with self.assertRaisesRegex(ValueError, "sampling/corruption protocol differs"):
            runner.validate_supplement(self.plan)

    def test_compatibility_does_not_bypass_data_checks(self):
        original = copy.deepcopy(self.package)
        for change, message in ((dict(source_splits_sha256="different"), "different original V3"),
                                (dict(conditions=[]), "not frozen"),
                                (dict(records=[]), "does not contain selected object")):
            with self.subTest(change=change):
                self.package = {**original, **change}
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, message):
                    runner.validate_supplement(self.plan)
        self.package = original
        self.write_manifest()
        (self.root / "object.npz").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "failed SHA256 verification"):
            runner.validate_supplement(self.plan)


if __name__ == "__main__":
    unittest.main()
