from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import audit_prior


class PriorAuditTest(unittest.TestCase):
    def test_copied_prior_artifacts_match_manifest(self) -> None:
        self.assertEqual(audit_prior.audit(), [])


if __name__ == "__main__":
    unittest.main()
