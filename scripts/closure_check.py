"""Generate a machine-readable closure manifest for the patched demo export."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
REG = ART / "model_registry" / "v2"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_tree_hash() -> tuple[str, int]:
    roots = [ROOT / "src", ROOT / "scripts", ROOT / "configs", ROOT / "tests", ROOT / "docs"]
    files = [ROOT / "README.md", ROOT / "ARCHITECTURE.md", ROOT / "pyproject.toml", ROOT / "requirements.txt"]
    for base in roots:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}:
                files.append(p)
    files = sorted(set(files), key=lambda p: p.relative_to(ROOT).as_posix())
    h = hashlib.sha256()
    for p in files:
        rel = p.relative_to(ROOT).as_posix().encode()
        h.update(rel + b"\0" + file_sha256(p).encode() + b"\n")
    return h.hexdigest(), len(files)


def parse_pytest_summary(path: Path):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"(?:=+\s*)?(?:(\d+) failed,\s*)?(\d+) passed(?:,\s*(\d+) skipped)?(?:,\s*(\d+) errors?)?", text)
    if not m:
        # Handle passed + skipped only ordering.
        m2 = re.search(r"(\d+) passed,\s*(\d+) skipped", text)
        if m2:
            return {"passed": int(m2.group(1)), "skipped": int(m2.group(2)), "failed": 0, "errors": 0}
        return {"summary_text": text[-1000:]}
    return {
        "failed": int(m.group(1) or 0),
        "passed": int(m.group(2) or 0),
        "skipped": int(m.group(3) or 0),
        "errors": int(m.group(4) or 0),
    }


def main():
    tree_hash, file_count = source_tree_hash()
    live_python = list((ROOT / "src" / "delay_intelligence").rglob("*.py"))
    forbidden_hits = []
    for p in live_python:
        text = p.read_text(encoding="utf-8", errors="replace").lower()
        for token in ["dummy_hash", "validated causal candidate"]:
            if token in text:
                forbidden_hits.append(f"{p.relative_to(ROOT)}:{token}")

    pages = sorted(p.name for p in (ROOT / "src" / "delay_intelligence" / "dashboard" / "pages").glob("*.py") if p.name != "__init__.py")
    expected_pages = ["01_executive.py", "02_shipment_explorer.py", "03_action_center.py", "04_analytics.py", "05_technical.py"]

    model_files = [
        "catboost_classifier.cbm", "lightgbm_q05.txt", "lightgbm_q50.txt", "lightgbm_q95.txt",
        "probability_calibration.json", "cqr_calibration.json", "feature_schema.json",
        "serving_validation.json", "metadata.json",
    ]
    model_hashes = {name: file_sha256(REG / name) for name in model_files}

    raw = ROOT / "data" / "raw" / "SCMS_Delivery_History_Dataset.csv"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_positioning": "Research-validated Decision Intelligence Prototype with a Production Roadmap",
        "source_tree_sha256": tree_hash,
        "source_tree_file_count": file_count,
        "git_commit": None,
        "git_note": "The uploaded source archive contained no .git history; no commit hash was invented.",
        "raw_scms": {"path": str(raw.relative_to(ROOT)), "sha256": file_sha256(raw), "label": "REAL DATA"},
        "serving_registry": {"path": "artifacts/model_registry/v2", "files_sha256": model_hashes, "label": "MODEL OUTPUT"},
        "dashboard_pages": pages,
        "dashboard_duplicates_removed": pages == expected_pages,
        "live_source_forbidden_claim_hits": forbidden_hits,
        "external_validation": {"DataCo": "NOT VALIDATED", "Olist": "NOT VALIDATED"},
        "causal_scope": "EXPLORATORY ONLY",
        "business_impact_scope": "SIMULATED SCENARIO",
        "verification": {
            "critical_suite": parse_pytest_summary(ART / "critical_test_suite.txt"),
            "reproducibility_check": (ART / "reproducibility_check.txt").read_text(encoding="utf-8", errors="replace").strip(),
            "full_legacy_suite_attempt": parse_pytest_summary(ART / "legacy_full_test_attempt.txt"),
            "full_legacy_suite_note": "Current execution image lacks pyarrow/fastparquet; many legacy research tests require Parquet. Optional DataCo/Olist tests are skipped because external datasets are intentionally not bundled.",
            "python": platform.python_version(),
            "pyarrow_available": importlib.util.find_spec("pyarrow") is not None,
            "streamlit_available": importlib.util.find_spec("streamlit") is not None,
        },
    }
    out = ART / "closure_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
