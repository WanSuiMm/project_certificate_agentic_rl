import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from function_swe_manifest import freeze_manifest, load_manifest, task_schedule
from tests.test_function_swe import toy_task


class FunctionSweManifestTests(unittest.TestCase):
    def test_freeze_and_detect_task_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_dir = root / "tasks"
            task_dir.mkdir()
            for index in range(64):
                task = toy_task()
                task["task_id"] = f"task-{index:02d}"
                task["probe_cases"] = [
                    {"args": [100 + case], "kwargs": {},
                     "expect": {"kind": "return", "value": 2 * (100 + case)}}
                    for case in range(256)
                ]
                (task_dir / f"{index:02d}.json").write_text(json.dumps(task), encoding="utf-8")
            manifest_path = root / "manifest.json"
            freeze_manifest(task_dir, manifest_path)
            manifest, tasks = load_manifest(manifest_path)
            self.assertEqual([entry["split"] for entry in manifest["tasks"]].count("train"), 48)
            self.assertEqual(len(tasks), 64)
            (task_dir / "00.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_manifest(manifest_path)

    def test_schedule_reproducible_and_four_distinct_tasks(self):
        ids = [f"task-{i}" for i in range(48)]
        first = task_schedule(ids, seed=0, updates=100, tasks_per_update=4)
        self.assertEqual(first, task_schedule(ids, seed=0, updates=100, tasks_per_update=4))
        self.assertEqual(len(first), 100)
        self.assertTrue(all(len(set(group)) == 4 for group in first))
        self.assertEqual(set(sum(first[:12], [])), set(ids))


if __name__ == "__main__":
    unittest.main()
