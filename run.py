#!/usr/bin/env python3
"""
run.py — Supply Chain Delay Intelligence System
================================================
Master launcher that runs every part of the project from a single entry point.

Usage:
    python run.py                   # Interactive menu
    python run.py --all             # Full pipeline (tests + all modules)
    python run.py --tests           # 659-test pytest suite
    python run.py --api             # FastAPI server  (http://127.0.0.1:8000/docs)
    python run.py --dashboard       # Streamlit dashboard (http://localhost:8501)
    python run.py --drift           # E6.5 drift detection
    python run.py --adaptive        # E7 adaptive conformal recalibration
    python run.py --cost            # E8 cost-sensitive holdout backtester
    python run.py --counterfactual  # E10 counterfactual policy evaluator
    python run.py --verify          # Closure reproducibility verification
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path

# ── Resolve project root ─────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent
VENV_PYTHON    = ROOT / ".venv" / "Scripts" / "python.exe"
VENV_PYTEST    = ROOT / ".venv" / "Scripts" / "pytest.exe"
VENV_UVICORN   = ROOT / ".venv" / "Scripts" / "uvicorn.exe"
VENV_STREAMLIT = ROOT / ".venv" / "Scripts" / "streamlit.exe"

PYTHON    = str(VENV_PYTHON)    if VENV_PYTHON.exists()    else sys.executable
PYTEST    = str(VENV_PYTEST)    if VENV_PYTEST.exists()    else "pytest"
UVICORN   = str(VENV_UVICORN)   if VENV_UVICORN.exists()   else "uvicorn"
STREAMLIT = str(VENV_STREAMLIT) if VENV_STREAMLIT.exists() else "streamlit"

# ── Colour helpers ───────────────────────────────────────────────────────────
RESET  = "\033[0m";  BOLD   = "\033[1m"
GREEN  = "\033[92m"; YELLOW = "\033[93m"
RED    = "\033[91m"; CYAN   = "\033[96m"
BLUE   = "\033[94m"

def c(text, colour):    return f"{colour}{text}{RESET}"
def header(title):
    bar = "=" * 64
    print(f"\n{c(bar, CYAN)}\n{c('  ' + title, BOLD + CYAN)}\n{c(bar, CYAN)}\n")
def step(msg):  print(f"  {c('▶', GREEN)} {msg}")
def ok(msg):    print(f"  {c('✓', GREEN)} {msg}")
def warn(msg):  print(f"  {c('⚠', YELLOW)} {msg}")
def fail(msg):  print(f"  {c('✗', RED)} {msg}")

# ── Subprocess helpers ────────────────────────────────────────────────────────
def run(cmd: list, cwd: Path = ROOT) -> int:
    print(f"\n  {c('$', BLUE)} {' '.join(str(x) for x in cmd)}\n")
    return subprocess.run(cmd, cwd=str(cwd)).returncode

def run_module(module: str) -> int:
    return run([PYTHON, "-m", module])

# ── Task functions ────────────────────────────────────────────────────────────

def task_tests() -> int:
    header("Test Suite — 659 tests")
    step("Running pytest …")
    code = run([PYTEST, "tests/", "--basetemp=scratch/pytest_temp", "-v", "--tb=short"])
    ok("All tests passed!") if code == 0 else fail(f"pytest exited with code {code}")
    return code

def task_drift() -> int:
    header("E6.5 — Drift Detection")
    step("Running chronological drift evaluation across CV folds …")
    code = run_module("delay_intelligence.drift.runner")
    ok("Artifacts → artifacts/drift/") if code == 0 else fail("Drift runner failed")
    return code

def task_adaptive() -> int:
    header("E7 — Adaptive Conformal Recalibration")
    step("Evaluating adaptive CQR on holdout …")
    code = run_module("delay_intelligence.adaptive_conformal.evaluator")
    ok("Artifacts → artifacts/adaptive_conformal/") if code == 0 else fail("Evaluator failed")
    return code

def task_cost() -> int:
    header("E8 — Cost-Sensitive Holdout Backtester")
    step("Running instance-dependent cost backtester …")
    code = run_module("delay_intelligence.cost_sensitive.holdout_evaluator")
    ok("Artifacts → artifacts/phase2/") if code == 0 else fail("Backtester failed")
    return code

def task_counterfactual() -> int:
    header("E10 — Counterfactual Policy Evaluator")
    step("Evaluating policies P0..P5 vs Oracle …")
    code = run_module("delay_intelligence.counterfactual.evaluator")
    ok("Artifacts → artifacts/phase2/") if code == 0 else fail("Evaluator failed")
    return code

def task_verify() -> int:
    header("Closure Reproducibility Verification")
    step("Verifying 36 SHA-256 cryptographic invariants …")
    code = run([PYTHON, "scripts/verify_closure_reproducibility.py"])
    ok("All 36 invariants verified — PASS") if code == 0 else fail("Verification failed")
    return code

def task_api():
    header("FastAPI REST API")
    step("Starting Uvicorn on http://127.0.0.1:8000")
    print(f"  {c('Swagger UI → http://127.0.0.1:8000/docs', YELLOW)}")
    print(f"  {c('Press Ctrl+C to stop', YELLOW)}\n")
    run([UVICORN, "delay_intelligence.api.main:app",
         "--host", "127.0.0.1", "--port", "8000", "--reload"])

def task_dashboard():
    header("Streamlit Control Tower Dashboard")
    step("Starting Streamlit on http://localhost:8501")
    print(f"  {c('Browser opens automatically', YELLOW)}")
    print(f"  {c('Press Ctrl+C to stop', YELLOW)}\n")
    run([STREAMLIT, "run", "src/delay_intelligence/dashboard/app.py"])

def task_all():
    header("Full Pipeline Run")
    stages = [
        ("Tests (659)",         task_tests),
        ("Drift Detection",     task_drift),
        ("Adaptive Conformal",  task_adaptive),
        ("Cost-Sensitive E8",   task_cost),
        ("Counterfactual E10",  task_counterfactual),
        ("Reproducibility",     task_verify),
    ]
    results = {}
    for name, fn in stages:
        t0 = time.time()
        code = fn()
        results[name] = ("PASS" if code == 0 else "FAIL", round(time.time() - t0, 1))

    header("Pipeline Summary")
    W = 28
    print(f"  {'Task':<{W}} {'Status':>6}  {'Time':>8}")
    print(f"  {'-'*W} {'------':>6}  {'--------':>8}")
    all_pass = True
    for name, (status, elapsed) in results.items():
        col = GREEN if status == "PASS" else RED
        print(f"  {name:<{W}} {c(status, col):>6}  {elapsed:>6.1f}s")
        if status != "PASS":
            all_pass = False
    print()
    ok("All pipeline stages passed!") if all_pass else fail("One or more stages failed")

# ── Interactive menu ──────────────────────────────────────────────────────────
MENU = [
    ("1", "Run Test Suite (659 tests)",              task_tests),
    ("2", "E6.5  Drift Detection",                   task_drift),
    ("3", "E7    Adaptive Conformal Recalibration",  task_adaptive),
    ("4", "E8    Cost-Sensitive Backtester",          task_cost),
    ("5", "E10   Counterfactual Policy Evaluator",   task_counterfactual),
    ("6", "Reproducibility Verification",            task_verify),
    ("7", "Launch FastAPI REST API  (blocking)",     task_api),
    ("8", "Launch Streamlit Dashboard (blocking)",   task_dashboard),
    ("9", "Run FULL pipeline (stages 1-6)",          task_all),
    ("0", "Exit",                                    None),
]

def interactive_menu():
    header("Supply Chain Delay Intelligence System — Launcher")
    print(f"  {c('Project root:', BOLD)} {ROOT}\n")
    for key, label, _ in MENU:
        print(f"  {c(f'[{key}]', CYAN)}  {label}")
    print()
    while True:
        choice = input(f"  {c('Select option >', BOLD + GREEN)} ").strip()
        for key, label, fn in MENU:
            if choice == key:
                if fn is None:
                    print(f"\n  {c('Goodbye!', CYAN)}\n")
                    sys.exit(0)
                fn()
                print()
                interactive_menu()
                return
        warn(f"Unknown option '{choice}' — choose from the menu above.")

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    if sys.platform == "win32":
        os.system("")   # enable ANSI colours in Windows console

    p = argparse.ArgumentParser(
        prog="python run.py",
        description="Supply Chain Delay Intelligence System — Master Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--all",            action="store_true", help="Run full pipeline")
    g.add_argument("--tests",          action="store_true", help="Run 659-test suite")
    g.add_argument("--drift",          action="store_true", help="E6.5 drift detection")
    g.add_argument("--adaptive",       action="store_true", help="E7 adaptive conformal")
    g.add_argument("--cost",           action="store_true", help="E8 cost-sensitive backtester")
    g.add_argument("--counterfactual", action="store_true", help="E10 counterfactual evaluator")
    g.add_argument("--verify",         action="store_true", help="Reproducibility verification")
    g.add_argument("--api",            action="store_true", help="Launch FastAPI server (blocking)")
    g.add_argument("--dashboard",      action="store_true", help="Launch Streamlit dashboard (blocking)")
    args = p.parse_args()

    dispatch = {
        "all": task_all, "tests": task_tests, "drift": task_drift,
        "adaptive": task_adaptive, "cost": task_cost,
        "counterfactual": task_counterfactual, "verify": task_verify,
        "api": task_api, "dashboard": task_dashboard,
    }
    for flag, fn in dispatch.items():
        if getattr(args, flag, False):
            fn()
            return

    interactive_menu()

if __name__ == "__main__":
    main()
