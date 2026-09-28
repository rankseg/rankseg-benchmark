from pathlib import Path

from rankseg_benchmark.nnunet.evidence import verify_evidence


def test_checked_in_full16_evidence_is_self_consistent():
    repository_root = Path(__file__).resolve().parents[2]
    assert verify_evidence(repository_root / "evidence/nnunet").is_file()
