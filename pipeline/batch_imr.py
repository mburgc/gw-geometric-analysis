#!/usr/bin/env python3
"""Granular IMR phase analysis for all events in the dataset.
Runs narrowband analysis on bands 30-500 Hz per event,
extracts IMR metrics (ISCO peak, merger drop, QNM oscillations)."""

import numpy as np, os, sys, json, warnings
warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib; matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(__file__))
from narrowband import analyze_narrowband
from gwpy.timeseries import TimeSeries as TS

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
CACHE = os.path.join(DATA, '_cache')
OUT = os.path.join(DATA, '_granular_imr')
os.makedirs(OUT, exist_ok=True)

# Bands: finer near ISCO/merger, coarser at high freq
BANDS = []
for low in range(30, 90, 20): BANDS.append((low, low+20))     # 30-90 Hz: 20 Hz res
for low in range(90, 210, 30): BANDS.append((low, low+30))    # 90-210 Hz: 30 Hz res
for low in range(210, 510, 50): BANDS.append((low, low+50))   # 210-500 Hz: 50 Hz res

# Get event list
events = sorted([d for d in os.listdir(DATA) if d.startswith('GW') and os.path.exists(os.path.join(DATA, d, 'pipeline_summary.json'))])
print(f'Events: {len(events)}, Bands per event: {len(BANDS)}, Total runs: {len(events)*len(BANDS)}')
print()

all_imr = {}

for ei, event in enumerate(events):
    # Load cached strain
    cache_path = os.path.join(CACHE, f'{event}_raw_strain.npz')
    if not os.path.exists(cache_path):
        print(f'[{ei+1}/{len(events)}] {event}: NO CACHE — skipping')
        continue
    
    print(f'[{ei+1}/{len(events)}] {event}: running {len(BANDS)} bands...', end=' ', flush=True)
    
    cache = np.load(cache_path, allow_pickle=True)
    gps = 1186741861  # default, will be overridden
    
    # Get GPS from summary
    sp = os.path.join(DATA, event, 'pipeline_summary.json')
    with open(sp) as f: s = json.load(f)
    gps = s['gps_center']
    
    h1_raw = cache['h1']; l1_raw = cache['l1']; v1_raw = cache['v1']
    cache.close()
    
    t0 = gps - 32//2
    h1_w = TS(h1_raw, t0=t0, sample_rate=4096).whiten(8,4)
    l1_w = TS(l1_raw, t0=t0, sample_rate=4096).whiten(8,4)
    v1_w = TS(v1_raw, t0=t0, sample_rate=4096).whiten(8,4)
    
    band_results = []
    for low, high in BANDS:
        outdir = os.path.join(OUT, f'{event}_{low}_{high}')
        os.makedirs(outdir, exist_ok=True)
        
        try:
            r = analyze_narrowband(event=event, gps_center=gps, band=(low, high),
                                   outdir=outdir, duration=32, fs=4096,
                                   strain_h1=h1_w.bandpass(low,high).value,
                                   strain_l1=l1_w.bandpass(low,high).value,
                                   strain_v1=v1_w.bandpass(low,high).value)
            eig = np.sort(r['eigvals_proj'])[::-1]
            eta = float(r['eta_proj'])
            r12 = eig[0]/max(eig[1],1e-12)
            r23 = eig[1]/max(eig[2],1e-12)
            band_results.append({'low': low, 'high': high, 'center': (low+high)/2,
                                 'eta': eta, 'r12': r12, 'r23': r23})
        except Exception as e:
            band_results.append({'low': low, 'high': high, 'center': (low+high)/2,
                                 'eta': None, 'r12': None, 'r23': None})
    
    # Extract IMR metrics
    valid = [b for b in band_results if b['eta'] is not None]
    if len(valid) < 5:
        print('FAIL (too few bands)')
        continue
    
    etas = np.array([b['eta'] for b in valid])
    centers = np.array([b['center'] for b in valid])
    r12s = np.array([b['r12'] for b in valid])
    r23s = np.array([b['r23'] for b in valid])
    
    # ISCO: peak η in 30-150 Hz
    isco_mask = (centers >= 30) & (centers <= 150)
    if np.sum(isco_mask) > 0:
        isco_idx = np.argmax(etas[isco_mask])
        isco_freq = centers[isco_mask][isco_idx]
        isco_eta = etas[isco_mask][isco_idx]
    else:
        isco_freq = None; isco_eta = None
    
    # Merger: max η drop in 80-200 Hz
    merger_mask = (centers >= 60) & (centers <= 200)
    if np.sum(merger_mask) >= 3:
        merger_etas = etas[merger_mask]
        diffs = np.diff(merger_etas)
        min_idx = np.argmin(diffs)
        merger_drop = diffs[min_idx]
        merger_drop_freq = centers[merger_mask][min_idx]
    else:
        merger_drop = None; merger_drop_freq = None
    
    # QNM: peak η in 200-500 Hz
    qnm_mask = (centers >= 200) & (centers <= 500)
    if np.sum(qnm_mask) > 0:
        qnm_idx = np.argmax(etas[qnm_mask])
        qnm_peak_freq = centers[qnm_mask][qnm_idx]
        qnm_peak_eta = etas[qnm_mask][qnm_idx]
        qnm_peak_r12 = r12s[qnm_mask][qnm_idx]
    else:
        qnm_peak_freq = None; qnm_peak_eta = None; qnm_peak_r12 = None
    
    # Oscillation amplitude (ringdown region)
    if np.sum(qnm_mask) >= 3:
        eta_amplitude = np.max(etas[qnm_mask]) - np.min(etas[qnm_mask])
    else:
        eta_amplitude = None
    
    # Number of Class A regions
    classes = []
    for b in valid:
        if b['eta'] > 0.35 and b['r12'] > 1.3 and b['r23'] > 1.3:
            classes.append('A')
        elif b['eta'] > 0.20:
            classes.append('B+')
        else:
            classes.append('C')
    
    # Count continuous Class A regions
    n_A_regions = 0
    in_A = False
    for c in classes:
        if c == 'A' and not in_A:
            n_A_regions += 1; in_A = True
        elif c != 'A':
            in_A = False
    
    # η range
    eta_max = np.max(etas)
    eta_min = np.min(etas)
    eta_std = np.std(etas)
    
    all_imr[event] = {
        'imr_isco_freq': round(isco_freq, 1) if isco_freq else None,
        'imr_isco_eta': round(isco_eta, 4) if isco_eta else None,
        'imr_merger_drop': round(merger_drop, 4) if merger_drop else None,
        'imr_merger_drop_freq': round(merger_drop_freq, 1) if merger_drop_freq else None,
        'imr_qnm_peak_freq': round(qnm_peak_freq, 1) if qnm_peak_freq else None,
        'imr_qnm_peak_eta': round(qnm_peak_eta, 4) if qnm_peak_eta else None,
        'imr_qnm_peak_r12': round(qnm_peak_r12, 2) if qnm_peak_r12 else None,
        'imr_eta_amplitude_ringdown': round(eta_amplitude, 4) if eta_amplitude else None,
        'imr_n_class_A_regions': n_A_regions,
        'imr_eta_max': round(eta_max, 4),
        'imr_eta_min': round(eta_min, 4),
        'imr_eta_std': round(eta_std, 4),
    }
    
    print(f'OK (ISCO={isco_freq}Hz, merger_drop={merger_drop}, QNM={qnm_peak_freq}Hz, n_A={n_A_regions})')

# Save results
out_path = os.path.join(DATA, '_granular_imr', 'imr_metrics.json')
with open(out_path, 'w') as f:
    json.dump(all_imr, f, indent=2)

completed = len(all_imr)
print(f'\nDone: {completed}/{len(events)} events processed.')
print(f'Metrics saved to: {out_path}')
