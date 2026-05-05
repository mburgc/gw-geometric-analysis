#!/usr/bin/env python3
# run_pipeline.py
# Full geometric analysis pipeline for a gravitational-wave event.

import os
import sys
import argparse
import json
import numpy as np

from narrowband import analyze_narrowband
from classify import classify_event, print_classification
from waveforms import reconstruct_and_compare

# Import gwpy for strain download
from gwpy.timeseries import TimeSeries


# ================================================================
# Event registry — GPS center times for known events
# ================================================================
EVENT_GPS = {
    "GW150914": 1126259462,
    "GW170729": 1185389807,
    "GW170809": 1186302519,
    "GW170814": 1186741861,
    "GW170817": 1187008882,
    "GW170818": 1187058327,
    "GW190412": 1239082262,
    "GW190521": 1242442967,
    "GW190814": 1249852257,
    "GW190924_021846": 1253326744,
}

# ================================================================
# Phase configuration — as defined in the paper
# ================================================================
PHASES = {
    "inspiral":         (30, 80),
    "post_inspiral":    (80, 150),
    "merger_ringdown":  (150, 200),
}

# ================================================================
# Default paths
# ================================================================
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEFAULT_DURATION = 32
DEFAULT_FS = 4096


def run(event, gps_center, data_dir=None, duration=DEFAULT_DURATION,
        fs=DEFAULT_FS, skip_download=False):
    """
    Run the full pipeline for one event.

    Parameters
    ----------
    event : str
        Event name.
    gps_center : int
        GPS center time.
    data_dir : str or None
        Output root directory. Defaults to ../data/ relative to this file.
    duration : int
        Time window in seconds.
    fs : int
        Sampling rate in Hz.
    skip_download : bool
        If True, skips phases whose npz already exists (for resuming).

    Returns
    -------
    dict
        Full pipeline results: phases, classification, waveform_comparison.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    # Per-event subdirectory to avoid overwriting
    event_dir = os.path.join(data_dir, event)
    os.makedirs(event_dir, exist_ok=True)
    cache_dir = os.path.join(data_dir, "_cache")
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 70)
    print(f"  GEOMETRIC GW PIPELINE — {event}")
    print(f"  GPS center: {gps_center}")
    print(f"  Data dir:   {event_dir}")
    print("=" * 70)

    # ----------------------------------------------------------
    # Phase 0: Download raw strain ONCE and cache
    # ----------------------------------------------------------
    cache_dir = os.path.join(data_dir, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{event}_raw_strain.npz")

    if os.path.exists(cache_path):
        print(f"\n📦 Cargando strain desde caché: {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        h1_raw_val = cached["h1"]
        l1_raw_val = cached["l1"]
        v1_raw_val = cached["v1"]
        cached.close()
    else:
        print(f"\n📡 Descargando strain de GWOSC ({event}, ±{duration//2}s, {fs} Hz)...")
        start = int(gps_center - duration // 2)
        end = int(gps_center + duration // 2)

        print("   H1 (Hanford)...")
        h1_raw_val = TimeSeries.fetch_open_data('H1', start, end, sample_rate=fs).value
        print("   L1 (Livingston)...")
        l1_raw_val = TimeSeries.fetch_open_data('L1', start, end, sample_rate=fs).value
        print("   V1 (Virgo)...")
        try:
            v1_raw_val = TimeSeries.fetch_open_data('V1', start, end, sample_rate=fs).value
        except Exception:
            print("   ⚠ V1 no disponible para esta época — usando ceros")
            v1_raw_val = np.zeros_like(h1_raw_val)

        print(f"   Guardando en caché: {cache_path}")
        np.savez_compressed(cache_path, h1=h1_raw_val, l1=l1_raw_val, v1=v1_raw_val)
        print(f"   ✓ {len(h1_raw_val)} muestras por detector")

    # Build TimeSeries objects once for whitening
    from gwpy.timeseries import TimeSeries as TS
    t0 = gps_center - duration // 2
    h1_ts = TS(h1_raw_val, t0=t0, sample_rate=fs)
    l1_ts = TS(l1_raw_val, t0=t0, sample_rate=fs)
    v1_ts = TS(v1_raw_val, t0=t0, sample_rate=fs)

    # Whitening is band-independent — do it once
    print("   Aplicando whitening (común a todas las bandas)...")
    h1_white = h1_ts.whiten(8, 4)
    l1_white = l1_ts.whiten(8, 4)
    v1_white = v1_ts.whiten(8, 4)

    # ----------------------------------------------------------
    # Phase 1: Three-band narrowband analysis
    # ----------------------------------------------------------
    results = {}

    for phase_name, band in PHASES.items():
        outdir = os.path.join(event_dir, phase_name)
        npz_path = os.path.join(outdir, "narrowband_improved.npz")

        if skip_download and os.path.exists(npz_path):
            print(f"\n⏭  {phase_name} ({band[0]}–{band[1]} Hz) — ya existe")
            data = np.load(npz_path, allow_pickle=True)
            results[phase_name] = {
                "eigvals_proj": data["eigvals_proj"],
                "eta_proj": data["eta_proj"],
                "Z_h_narrow": data["Z_h_narrow"],
                "Z_l_narrow": data["Z_l_narrow"],
                "Z_v_narrow": data["Z_v_narrow"],
                "times": data["times"],
                "freqs": data["freqs"],
                "npz_path": npz_path,
            }
            continue

        print(f"\n{'─' * 60}")
        print(f"  PHASE: {phase_name}  |  Band: {band[0]}–{band[1]} Hz")
        print(f"{'─' * 60}")

        # Bandpass the pre-whitened data
        h1_proc = h1_white.bandpass(band[0], band[1]).value
        l1_proc = l1_white.bandpass(band[0], band[1]).value
        v1_proc = v1_white.bandpass(band[0], band[1]).value

        r = analyze_narrowband(
            event=event,
            gps_center=gps_center,
            band=band,
            outdir=outdir,
            duration=duration,
            fs=fs,
            strain_h1=h1_proc,
            strain_l1=l1_proc,
            strain_v1=v1_proc,
        )
        results[phase_name] = r

    # ----------------------------------------------------------
    # Phase 2: Geometric classification
    # ----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  GEOMETRIC CLASSIFICATION")
    print(f"{'=' * 60}")

    classification = classify_event(results)
    print_classification(classification)
    classification["event"] = event

    # ----------------------------------------------------------
    # Phase 3: Waveform reconstruction & comparison
    # ----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  WAVEFORM RECONSTRUCTION & COMPARISON")
    print(f"{'=' * 60}")

    waveform_metrics = reconstruct_and_compare(
        results=results,
        event=event,
        gps_center=gps_center,
        outdir=event_dir,
        duration=duration,
        show_plot=False,
    )

    # ----------------------------------------------------------
    # Save summary
    # ----------------------------------------------------------
    summary = {
        "event": event,
        "gps_center": gps_center,
        "phases": {
            name: {
                "band": list(PHASES[name]),
                "eta_proj": float(results[name]["eta_proj"]),
                "eigvals_proj": results[name]["eigvals_proj"].tolist(),
                "npz_path": results[name].get("npz_path", ""),
            }
            for name in PHASES
        },
        "classification": classification,
        "waveform_comparison": waveform_metrics,
    }

    summary_path = os.path.join(event_dir, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n✓ Summary saved to: {summary_path}")

    # ----------------------------------------------------------
    # Final report
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE — {event}")
    print(f"  Class: {classification['class']}")
    print(f"  η (merger/ringdown): {classification['eta']:.3f}")
    print(f"  λ1/λ2: {classification['lambda1_lambda2']:.2f}")
    print(f"  Global waveform correlation: {waveform_metrics['global_waveform_correlation']:.4f}")
    print("=" * 70)

    return {
        "phases": results,
        "classification": classification,
        "waveform_comparison": waveform_metrics,
    }


# ================================================================
# CLI
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Template-Free Geometric Analysis of GW Events"
    )
    parser.add_argument("--event", default="GW170814",
                        help="Event name (default: GW170814)")
    parser.add_argument("--gps", type=int, default=None,
                        help="GPS center time (auto-detected for known events)")
    parser.add_argument("--data-dir", default=None,
                        help="Output directory (default: ../data)")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                        help=f"Time window in seconds (default: {DEFAULT_DURATION})")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip phases whose npz already exists")

    args = parser.parse_args()

    event = args.event
    gps_center = args.gps or EVENT_GPS.get(event)

    if gps_center is None:
        # Auto-detect from GWOSC API
        try:
            import requests
            url = 'https://gwosc.org/eventapi/json/GWTC/'
            data = requests.get(url, timeout=15).json()['events']
            for key, info in data.items():
                if info.get('commonName', '') == event:
                    gps_center = int(info.get('GPS', 0))
                    print(f"Auto-detected GPS: {gps_center}")
                    break
        except Exception:
            pass

    if gps_center is None:
        print(f"Unknown event: {event}")
        print("Provide --gps to specify the GPS center time manually.")
        sys.exit(1)

    run(
        event=event,
        gps_center=gps_center,
        data_dir=args.data_dir,
        duration=args.duration,
        skip_download=args.skip_existing,
    )


if __name__ == "__main__":
    main()
