#!/usr/bin/env python3
# narrowband_per_detector.py
# EXPERIMENTAL: Geometric analysis per individual detector (H1, L1, V1 separately).
# Unlike narrowband.py which stacks detectors, this analyzes each detector independently.
# Useful for: single-detector events, glitch identification, detector quality comparison.

import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})

from gwpy.timeseries import TimeSeries
from scipy.signal import stft


def analyze_per_detector(event, gps_center, band, outdir,
                         strain_h1=None, strain_l1=None, strain_v1=None,
                         detectors_active=None,
                         duration=32, fs=4096, plot=False):
    """
    Run geometric analysis PER DETECTOR (independently).

    Returns a dict with per-detector η, λ ratios, classification, and
    a consensus flag indicating whether all active detectors agree.

    Parameters
    ----------
    detectors_active : list or None
        Which detectors have real data, e.g. ['H1', 'L1']. Others use zeros.
        If None, assumes all 3 are active.
    """
    BAND_NARROW = band; DURATION = duration; FS = fs
    N_PER_SEG = int(FS * 0.25); N_OVERLAP = int(N_PER_SEG * 0.5)
    EVENT = event; GPS_CENTER = gps_center
    os.makedirs(outdir, exist_ok=True)

    if detectors_active is None:
        detectors_active = ['H1', 'L1', 'V1']

    det_data = {'H1': strain_h1, 'L1': strain_l1, 'V1': strain_v1}

    # ---------- PROCESS EACH DETECTOR ----------
    results_per_det = {}
    for det_name in ['H1', 'L1', 'V1']:
        arr = det_data[det_name]
        if arr is None or np.all(arr == 0):
            results_per_det[det_name] = {
                'eta': None, 'lambda1': None, 'lambda2': None,
                'r12': None, 'r23': None, 'class': 'N/A',
                'active': False, 'snr': 0,
            }
            continue

        is_active = det_name in detectors_active

        # STFT
        freqs, times, Z = stft(arr, fs=FS, nperseg=N_PER_SEG,
                               noverlap=N_OVERLAP, window='hamming')
        freq_mask = (freqs >= BAND_NARROW[0]) & (freqs <= BAND_NARROW[1])
        Z_narrow = Z[freq_mask, :]
        freqs_narrow = freqs[freq_mask]

        # Energy and active windows
        energy = np.sum(np.abs(Z_narrow)**2, axis=0)
        combined_energy = energy  # single detector, no combination
        thr = np.percentile(combined_energy, 75)
        active_windows = combined_energy > thr

        snr = np.max(energy) / (np.median(energy) + 1e-12)

        if np.sum(active_windows) < 3:
            results_per_det[det_name] = {
                'eta': None, 'lambda1': None, 'lambda2': None,
                'r12': None, 'r23': None, 'class': 'N/A',
                'active': is_active, 'snr': snr,
                'n_active_windows': int(np.sum(active_windows)),
            }
            continue

        # Spectral operator for this detector alone
        Z_use = Z_narrow[:, active_windows]
        C = (Z_use @ np.conj(Z_use.T)) / Z_use.shape[1]

        # Off-source projection (single-detector version)
        off_windows = ~active_windows
        Z_off = Z_narrow[:, off_windows]
        C_instr = Z_off @ np.conj(Z_off.T) / max(Z_off.shape[1], 1)

        # Eigendecomposition
        w_instr, v_instr = np.linalg.eigh(C_instr)
        idx = np.argsort(w_instr)[::-1]
        w_instr = w_instr[idx]
        v_instr = v_instr[:, idx]

        # Project dominant instrumental modes
        k_instr = min(2, len(w_instr))
        P_perp = np.eye(v_instr.shape[0], dtype=complex)
        for i in range(k_instr):
            vi = v_instr[:, i].reshape(-1, 1)
            P_perp -= vi @ vi.conj().T

        Z_proj = P_perp @ Z_use
        C_proj = Z_proj @ Z_proj.conj().T / Z_proj.shape[1]

        w_proj, _ = np.linalg.eigh(C_proj)
        w_proj = np.sort(w_proj)[::-1]

        eigvals = w_proj[:min(3, len(w_proj))]
        if len(eigvals) >= 3:
            eta = eigvals[0] / np.sum(eigvals)
            r12 = eigvals[0] / max(eigvals[1], 1e-12)
            r23 = eigvals[1] / max(eigvals[2], 1e-12)
        elif len(eigvals) == 2:
            eta = eigvals[0] / np.sum(eigvals)
            r12 = eigvals[0] / max(eigvals[1], 1e-12)
            r23 = None
        else:
            eta = 1.0; r12 = None; r23 = None

        # Classify per-detector
        if eta is not None:
            if eta > 0.35 and r12 and r23 and r12 > 1.3 and r23 and r23 > 1.3:
                cls = 'A'
            elif eta > 0.20:
                cls = 'B+'
            else:
                cls = 'C'
        else:
            cls = 'N/A'

        results_per_det[det_name] = {
            'eta': eta, 'lambda1': eigvals[0] if len(eigvals) > 0 else None,
            'lambda2': eigvals[1] if len(eigvals) > 1 else None,
            'r12': r12, 'r23': r23, 'class': cls,
            'active': is_active, 'snr': snr,
            'n_active_windows': int(np.sum(active_windows)),
        }

    # ---------- CONSENSUS CHECK ----------
    active_results = {k: v for k, v in results_per_det.items() if v['active'] and v['eta'] is not None}

    if len(active_results) >= 2:
        classes = [v['class'] for v in active_results.values()]
        etas = [v['eta'] for v in active_results.values()]
        consensus_class = max(set(classes), key=classes.count)
        consensus_eta = np.mean(etas)
        consensus = (len(set(classes)) == 1)  # all same class
    elif len(active_results) == 1:
        det = list(active_results.keys())[0]
        consensus_class = active_results[det]['class']
        consensus_eta = active_results[det]['eta']
        consensus = None  # single detector, can't verify
    else:
        consensus_class = 'N/A'
        consensus_eta = None
        consensus = False

    return {
        'event': EVENT,
        'band': BAND_NARROW,
        'per_detector': results_per_det,
        'consensus_class': consensus_class,
        'consensus_eta': consensus_eta,
        'consensus': consensus,
        'n_active_detectors': len(active_results),
        'detectors_available': detectors_active,
    }


def print_per_detector_report(results):
    """Pretty-print the per-detector analysis report."""
    pd = results['per_detector']
    print()
    print(f"═══ Per-Detector Geometric Analysis: {results['event']} ({results['band'][0]}-{results['band'][1]} Hz) ═══")
    print(f"  Active detectors: {results['detectors_available']}")
    print()
    print(f"  {'Detector':<10s} {'Active':<8s} {'η':>8s} {'λ1/λ2':>8s} {'Class':<6s} {'SNR':>7s} {'Windows':>8s}")
    print(f"  {'-'*60}")
    for det in ['H1', 'L1', 'V1']:
        d = pd[det]
        eta_str = f"{d['eta']:.4f}" if d['eta'] is not None else 'N/A'
        r12_str = f"{d['r12']:.2f}" if d['r12'] is not None else 'N/A'
        snr_str = f"{d['snr']:.1f}" if d['snr'] else 'N/A'
        win_str = str(d.get('n_active_windows', '?'))
        active_str = '✅' if d['active'] else '❌'
        print(f"  {det:<10s} {active_str:<8s} {eta_str:>8s} {r12_str:>8s} {d['class']:<6s} {snr_str:>7s} {win_str:>8s}")
    print()
    print(f"  CONSENSUS: Class={results['consensus_class']}, η_mean={results['consensus_eta']}")
    if results['consensus'] is True:
        print(f"  ✅ All active detectors agree on classification.")
    elif results['consensus'] is False:
        print(f"  ⚠️  Detectors DISAGREE — possible glitch in one detector.")
    else:
        print(f"  ℹ️  Single detector — cannot verify consensus.")
    print()


if __name__ == "__main__":
    # Quick test
    print("Per-detector analysis module loaded.")
    print("Usage: from narrowband_per_detector import analyze_per_detector, print_per_detector_report")
