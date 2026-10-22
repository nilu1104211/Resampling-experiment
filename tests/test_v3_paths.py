"""Relocation must preserve source identity and official point coordinates."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from adaptive_resampling.core import input_transform
from adaptive_resampling.v3_dataset import OfficialPU1KDataset
from adaptive_resampling.v3_paths import SourcePaths


class RelocationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "v3"
        (self.root / "metadata").mkdir(parents=True)
        self.write_json("generation_config.json", {"raw_root": "D:/old/data"})

    def write_json(self, name, value):
        (self.root / "metadata" / name).write_text(json.dumps(value), encoding="utf-8")

    def test_windows_sources_map_to_the_same_relative_file(self):
        raw = Path(self.temporary.name) / "raw"
        with patch.dict(os.environ, {"V3_RAW_ROOT": str(raw)}):
            resolver = SourcePaths(self.root)
            for stored in ["D:/old/data/PU1K/shape.xyz", r"d:\old\data\PU1K\shape.xyz"]:
                self.assertEqual(Path(resolver.resolve(stored)), raw / "PU1K/shape.xyz")
            # A similarly named directory must not be remapped.
            self.assertEqual(resolver.resolve("D:/old/database/shape.xyz"), "D:/old/database/shape.xyz")

    def test_relative_sources_and_unmapped_absolute_sources(self):
        with patch.dict(os.environ, {"V3_RAW_ROOT": ""}):
            resolver = SourcePaths(self.root)
            self.assertEqual(Path(resolver.resolve("sources/shape.off")), self.root / "sources/shape.off")
            self.assertEqual(resolver.resolve("/original/shape.off"), "/original/shape.off")
            self.assertEqual(resolver.resolve(r"D:\old\data\shape.off"), "D:/old/data/shape.off")

    def test_posix_prefixes_remain_case_sensitive(self):
        self.write_json("generation_config.json", {"raw_root": "/old/data"})
        with patch.dict(os.environ, {"V3_RAW_ROOT": str(self.root / "raw")}):
            resolver = SourcePaths(self.root)
            self.assertEqual(Path(resolver.resolve("/old/data/shape.off")), self.root / "raw/shape.off")
            self.assertEqual(resolver.resolve("/Old/data/shape.off"), "/Old/data/shape.off")

    def test_official_loader_reads_relocated_pairs_without_changing_geometry(self):
        raw = Path(self.temporary.name) / "raw"
        raw.mkdir()
        rng = np.random.default_rng(31)
        points, target = rng.normal(size=(2048, 3)), rng.normal(size=(8192, 3))
        np.savetxt(raw / "shape.xyz", points)
        np.savetxt(raw / "target.xyz", target)
        self.write_json("official_benchmark.json", {"pairs": [{
            "sample_id": "pu1k_official/shape", "input_path": r"D:\old\data\shape.xyz",
            "target_path": r"D:\old\data\target.xyz",
        }]})
        self.write_json("dataset_splits.json", {"pu1k_test": [{
            "object_id": "pu1k/shape", "path": r"D:\old\data\shape.off",
            "sha256": "same-mesh", "geometry_id": "same-geometry",
        }]})
        manifest = (self.root / "metadata/official_benchmark.json").read_bytes()
        with patch.dict(os.environ, {"V3_RAW_ROOT": str(raw)}):
            sample = OfficialPU1KDataset(self.root, include_reference_metadata=True)[0]
        expected_input, expected_target, _, _ = input_transform(points, target)
        np.testing.assert_array_equal(sample["input_xyz"], expected_input)
        np.testing.assert_array_equal(sample["gt_xyz"], expected_target)
        self.assertEqual(sample["object_id"], "pu1k/shape")
        self.assertEqual(sample["sample_id"], "pu1k_official/shape")
        self.assertEqual(Path(sample["reference"]["mesh_path"]), raw / "shape.off")
        self.assertEqual((self.root / "metadata/official_benchmark.json").read_bytes(), manifest)


if __name__ == "__main__":
    unittest.main()
