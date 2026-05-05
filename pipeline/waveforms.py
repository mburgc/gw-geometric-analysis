# waveforms.py
# Geometric waveform reconstruction from coherent narrowband modes.
# Refactored from the original script — now callable as a function.

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.interpolate import interp1d
from gwpy.timeseries import TimeSeries


# ==================================================
# Utilities
# ==================================================
def normalize(x):
    m = np.max(np.abs(x))
    return x / m if m > 0 else x


def envelope(x):
    return np.abs(hilbert(x))


def reconstruct_from_Z(Z):
    """Geometric reconstruction from coherent modes."""
    h = np.real(np.sum(Z, axis=0))
    return normalize(h)


def energy(x):
    return x**2


# ==================================================
# Main reconstruction function
# ==================================================
def reconstruct_and_compare(results, event="GW170814", gps_center=1186741861,
                            outdir=None, duration=32, show_plot=True):
    """
    Reconstruct geometric waveform from 3-phase narrowband results
    and compare with official LIGO strain.

    Parameters
    ----------
    results : dict
        Phase-level results from analyze_narrowband().
        Must contain keys: 'inspiral', 'post_inspiral', 'merger_ringdown'.
        Each phase dict must have: 'Z_h_narrow', 'Z_l_narrow', 'Z_v_narrow', 'times'.
    event : str
        Event name (default "GW170814").
    gps_center : int
        GPS center time.
    outdir : str or None
        If provided, saves the comparison plot to this directory.
    duration : int
        Time window for fetching LIGO strain (default 32 s).
    show_plot : bool
        Whether to display the plot interactively.

    Returns
    -------
    dict
        Comparison metrics: global correlation, envelope correlation, RMS residuals,
        per-phase statistics.
    """
    phase_order = ['inspiral', 'post_inspiral', 'merger_ringdown']
    phase_labels = ['Inspiral', 'Post-inspiral', 'Merger / Ringdown']

    h_geom = []
    t_geom = None

    for phase_key, label in zip(phase_order, phase_labels):
        r = results[phase_key]

        Z_h = r["Z_h_narrow"]
        Z_l = r["Z_l_narrow"]
        Z_v = r["Z_v_narrow"]

        if t_geom is None:
            t_geom = r["times"]

        h_net = (
            reconstruct_from_Z(Z_h) +
            reconstruct_from_Z(Z_l) +
            reconstruct_from_Z(Z_v)
        ) / 3.0

        h_geom.append(h_net)

    h_geom = normalize(np.sum(h_geom, axis=0))

    # ==================================================
    # Load official LIGO strain
    # ==================================================
    window = duration
    strain = TimeSeries.fetch_open_data(
        "H1",
        gps_center - window,
        gps_center + window
    ).bandpass(30, 80)

    t_strain = strain.times.value
    h_strain = normalize(strain.value)

    # ==================================================
    # Interpolate official strain onto geometric timeline
    # ==================================================
    t_geom_abs = t_geom - t_geom.mean() + gps_center

    interp = interp1d(
        t_strain,
        h_strain,
        bounds_error=False,
        fill_value=0.0
    )

    h_strain_interp = normalize(interp(t_geom_abs))
    t_plot = t_geom_abs - gps_center

    # ==================================================
    # Envelope & energy diagnostics
    # ==================================================
    env_geom = envelope(h_geom)
    env_strain = envelope(h_strain_interp)

    E_geom = energy(h_geom)
    E_strain = energy(h_strain_interp)

    env_geom = normalize(env_geom)
    env_strain = normalize(env_strain)
    E_geom = normalize(E_geom)
    E_strain = normalize(E_strain)

    # ==================================================
    # Residual
    # ==================================================
    residual_env = env_strain - env_geom
    residual_E = E_strain - E_geom

    # ==================================================
    # Phase markers
    # ==================================================
    n_phases = len(phase_order)
    t_start = t_plot[0]
    t_end = t_plot[-1]
    phase_times = np.linspace(t_start, t_end, n_phases + 1)
    phase_colors = ["tab:blue", "tab:orange", "tab:red"]

    # ==================================================
    # Global and per-phase statistics
    # ==================================================
    global_corr = np.corrcoef(h_geom, h_strain_interp)[0, 1]
    global_env_corr = np.corrcoef(env_geom, env_strain)[0, 1]
    global_rms_env = np.sqrt(np.mean(residual_env**2))
    global_rms_energy = np.sqrt(np.mean(residual_E**2))

    per_phase_stats = []
    for i in range(n_phases):
        mask = (t_plot >= phase_times[i]) & (t_plot <= phase_times[i + 1])
        corr = np.corrcoef(h_geom[mask], h_strain_interp[mask])[0, 1]
        env_corr = np.corrcoef(env_geom[mask], env_strain[mask])[0, 1]
        rms_env = np.sqrt(np.mean(residual_env[mask]**2))
        rms_E = np.sqrt(np.mean(residual_E[mask]**2))
        per_phase_stats.append({
            "phase": phase_labels[i],
            "waveform_correlation": float(corr),
            "envelope_correlation": float(env_corr),
            "envelope_rms_residual": float(rms_env),
            "energy_rms_residual": float(rms_E),
        })

    # ==================================================
    # Visualization
    # ==================================================
    plt.figure(figsize=(14, 12))

    # Waveforms
    plt.subplot(4, 1, 1)
    plt.plot(t_plot, h_strain_interp, label="LIGO strain (H1)", color="steelblue", alpha=0.7)
    plt.plot(t_plot, h_geom, label="Geometric reconstruction", color="black", alpha=0.8)
    plt.title("Waveform comparison (reference)")
    plt.ylabel("Normalized strain")
    plt.legend()
    for i, pt in enumerate(phase_times[1:-1]):
        plt.axvline(pt, color=phase_colors[i + 1], linestyle="--", alpha=0.6, linewidth=1.2)
    for i in range(n_phases):
        t_mid = 0.5 * (phase_times[i] + phase_times[i + 1])
        plt.text(t_mid, 0.85, phase_labels[i], color=phase_colors[i],
                 ha="center", va="center", fontsize=10, alpha=0.9,
                 transform=plt.gca().get_xaxis_transform())
    plt.grid(alpha=0.3)

    # Envelope
    plt.subplot(4, 1, 2)
    plt.plot(t_plot, env_strain, label="Envelope (real)", color="steelblue")
    plt.plot(t_plot, env_geom, label="Envelope (geometric)", color="black")
    plt.title("Amplitude envelope (phase-invariant)")
    plt.ylabel("Normalized envelope")
    plt.legend()
    plt.grid(alpha=0.3)

    # Energy
    plt.subplot(4, 1, 3)
    plt.plot(t_plot, E_strain, label="Energy (real)", color="steelblue")
    plt.plot(t_plot, E_geom, label="Energy (geometric)", color="black")
    plt.title("Instantaneous energy")
    plt.ylabel("Normalized energy")
    plt.legend()
    plt.grid(alpha=0.3)

    # Residual
    plt.subplot(4, 1, 4)
    plt.plot(t_plot, residual_env, label="Envelope residual", color="crimson")
    plt.plot(t_plot, residual_E, label="Energy residual", color="darkred", alpha=0.6)
    plt.title("Structural residuals (geometry vs detector)")
    plt.xlabel("Time relative to merger (s)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    if outdir:
        import os
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, 'waveform_comparison.png'), dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()

    # ==================================================
    # Print summary
    # ==================================================
    print("\n=== WAVEFORM RECONSTRUCTION SUMMARY ===")
    print(f"Global waveform correlation: {global_corr:.4f}")
    print(f"Global envelope correlation: {global_env_corr:.4f}")
    print(f"Global envelope RMS residual: {global_rms_env:.4f}")
    print(f"Global energy RMS residual: {global_rms_energy:.4f}")
    print("\nPer-phase:")
    for s in per_phase_stats:
        print(f"  {s['phase']}: corr={s['waveform_correlation']:.4f}, "
              f"env_corr={s['envelope_correlation']:.4f}")

    return {
        "global_waveform_correlation": float(global_corr),
        "global_envelope_correlation": float(global_env_corr),
        "global_envelope_rms_residual": float(global_rms_env),
        "global_energy_rms_residual": float(global_rms_energy),
        "per_phase": per_phase_stats,
    }
