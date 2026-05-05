#!/usr/bin/env python3
"""Batch runner: process ALL O2+O3 confident events through the geometric pipeline."""
import subprocess, sys, os, time

PIPELINE = os.path.join(os.path.dirname(__file__), "run_pipeline.py")
VENV_PYTHON = "/home/cocaik/gw_venv/bin/python"

EVENTS = [
    "GW170608", "GW170823",
    "GW190403_051519", "GW190408_181802", "GW190413_052954", "GW190413_134308",
    "GW190421_213856", "GW190425", "GW190426_190642", "GW190503_185404",
    "GW190512_180714", "GW190513_205428", "GW190514_065416", "GW190517_055101",
    "GW190519_153544", "GW190521_074359", "GW190527_092055", "GW190602_175927",
    "GW190620_030421", "GW190630_185205", "GW190701_203306", "GW190706_222641",
    "GW190707_093326", "GW190708_232457", "GW190719_215514", "GW190720_000836",
    "GW190725_174728", "GW190727_060333", "GW190728_064510", "GW190731_140936",
    "GW190803_022701", "GW190805_211137", "GW190828_063405", "GW190828_065509",
    "GW190910_112807", "GW190915_235702", "GW190916_200658", "GW190917_114630",
    "GW190925_232845", "GW190926_050336", "GW190929_012149", "GW190930_133541",
    "GW191103_012549", "GW191105_143521", "GW191109_010717", "GW191113_071753",
    "GW191126_115259", "GW191127_050227", "GW191129_134029", "GW191204_110529",
    "GW191204_171526", "GW191215_223052", "GW191216_213338", "GW191219_163120",
    "GW191222_033537", "GW191230_180458", "GW200112_155838", "GW200115_042309",
    "GW200128_022011", "GW200129_065458", "GW200202_154313", "GW200208_130117",
    "GW200208_222617", "GW200209_085452", "GW200210_092254", "GW200216_220804",
    "GW200219_094415", "GW200220_061928", "GW200220_124850", "GW200224_222234",
    "GW200225_060421", "GW200302_015811", "GW200306_093714", "GW200308_173609",
    "GW200311_115853", "GW200316_215756", "GW200322_091133",
]

total = len(EVENTS)
results = {"ok": [], "fail": []}
start_all = time.time()

for i, event in enumerate(EVENTS):
    print(f"\n{'█'*70}")
    print(f"█  [{i+1}/{total}] EVENT: {event}")
    print(f"{'█'*70}")
    t0 = time.time()

    ret = subprocess.run(
        [VENV_PYTHON, "-u", PIPELINE, "--event", event, "--skip-existing"],
        cwd=os.path.dirname(PIPELINE),
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=False,
    )

    elapsed = time.time() - t0
    if ret.returncode == 0:
        print(f"\n✓ {event} OK ({elapsed:.0f}s)")
        results["ok"].append(event)
    else:
        print(f"\n✗ {event} FAILED (exit {ret.returncode}, {elapsed:.0f}s)")
        results["fail"].append(event)

total_time = time.time() - start_all
print("\n" + "=" * 70)
print(f"BATCH COMPLETE — {total_time/60:.1f} min")
print(f"  OK:     {len(results['ok'])}/{total}")
if results["fail"]:
    print(f"  FAILED: {', '.join(results['fail'])}")
print("=" * 70)

sys.exit(1 if results["fail"] else 0)
