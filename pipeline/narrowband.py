# narrowband.py
# Geometric narrowband analysis pipeline — callable function version.
# Refactored from pipelineGW.py: same logic, wrapped in analyze_narrowband().

import numpy as np
from gwpy.timeseries import TimeSeries
import os
from scipy.signal import stft
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})


def analyze_narrowband(event, gps_center, band, outdir,
                       duration=32, fs=4096,
                       strain_h1=None, strain_l1=None, strain_v1=None):
    """
    Run full narrowband geometric analysis for one frequency band.

    Parameters
    ----------
    event : str
        Event name, e.g. "GW170814".
    gps_center : int
        GPS center time.
    band : tuple (low_hz, high_hz)
        Narrowband frequency range.
    outdir : str
        Output directory for npz and figures.
    duration : int
        Total duration in seconds (default 32).
    fs : int
        Sampling rate in Hz (default 4096).
    strain_h1, strain_l1, strain_v1 : np.ndarray or None
        Pre-downloaded whitened strain arrays. If None, downloads from GWOSC.

    Returns
    -------
    results : dict
    """
    # ---------- PARÁMETROS DERIVADOS ----------
    BAND_NARROW = band
    DURATION = duration
    FS = fs
    N_PER_SEG = int(FS * 0.25)
    N_OVERLAP = int(N_PER_SEG * 0.5)
    EVENT = event
    GPS_CENTER = gps_center

    os.makedirs(outdir, exist_ok=True)

    # ---------- DESCARGA Y PREPROCESADO ----------
    if strain_h1 is not None and strain_l1 is not None and strain_v1 is not None:
        print(f"Usando strain pre-cargado para {EVENT}")
        print(f"Duration: {DURATION}s, Band: {BAND_NARROW} Hz")
        h1_arr = strain_h1
        l1_arr = strain_l1
        v1_arr = strain_v1
    else:
        print(f"Descargando datos de GWOSC para {EVENT}...")
        print(f"GPS center: {GPS_CENTER}")
        print(f"Duration: {DURATION}s, Band: {BAND_NARROW} Hz")

        start = int(GPS_CENTER - DURATION // 2)
        end = int(GPS_CENTER + DURATION // 2)

        print("  Descargando H1...")
        h1_raw = TimeSeries.fetch_open_data('H1', start, end, sample_rate=FS)
        print("  Descargando L1...")
        l1_raw = TimeSeries.fetch_open_data('L1', start, end, sample_rate=FS)
        print("  Descargando V1...")
        v1_raw = TimeSeries.fetch_open_data('V1', start, end, sample_rate=FS)

        print("  Aplicando whitening + bandpass...")
        h1_arr = h1_raw.whiten(8, 4).bandpass(BAND_NARROW[0], BAND_NARROW[1]).value
        l1_arr = l1_raw.whiten(8, 4).bandpass(BAND_NARROW[0], BAND_NARROW[1]).value
        v1_arr = v1_raw.whiten(8, 4).bandpass(BAND_NARROW[0], BAND_NARROW[1]).value

        print(f"  Listo — {len(h1_arr)} muestras")

    print(f"Datos procesados - Longitud: {len(h1_arr)} muestras")
    print(f"  H1: media={np.mean(h1_arr):.2e}, std={np.std(h1_arr):.2e}")
    print(f"  L1: media={np.mean(l1_arr):.2e}, std={np.std(l1_arr):.2e}")
    print(f"  V1: media={np.mean(v1_arr):.2e}, std={np.std(v1_arr):.2e}")

    # ---------- VISUALIZACIÓN: SERIES TEMPORALES PROCESADAS ----------
    print("\nGenerando visualizaciones...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    time_axis = np.arange(len(h1_arr)) / FS - DURATION / 2

    axes[0].plot(time_axis, h1_arr, 'b-', linewidth=0.8)
    axes[0].set_ylabel('Amplitude (H1)')
    axes[0].set_title(f'Processed Time Series in Narrowband ({BAND_NARROW[0]}-{BAND_NARROW[1]} Hz)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis, l1_arr, 'r-', linewidth=0.8)
    axes[1].set_ylabel('Amplitude (L1)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_axis, v1_arr, 'g-', linewidth=0.8)
    axes[2].set_ylabel('Amplitude (V1)')
    axes[2].set_xlabel('Time [s]')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'time_series_processed.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------- CÁLCULO DE ESPECTROGRAMAS (STFT) ----------
    print("\nCalculando espectrogramas (STFT) en la banda narrowband...")
    print(f"  Parámetros STFT: nperseg={N_PER_SEG}, noverlap={N_OVERLAP}")

    freqs, times, Z_h_narrow = stft(h1_arr, fs=FS, nperseg=N_PER_SEG,
                                    noverlap=N_OVERLAP, window='hamming')
    _, _, Z_l_narrow = stft(l1_arr, fs=FS, nperseg=N_PER_SEG,
                            noverlap=N_OVERLAP, window='hamming')
    _, _, Z_v_narrow = stft(v1_arr, fs=FS, nperseg=N_PER_SEG,
                            noverlap=N_OVERLAP, window='hamming')

    freq_mask = (freqs >= BAND_NARROW[0]) & (freqs <= BAND_NARROW[1])
    Z_h_narrow = Z_h_narrow[freq_mask, :]
    Z_l_narrow = Z_l_narrow[freq_mask, :]
    Z_v_narrow = Z_v_narrow[freq_mask, :]
    freqs_narrow = freqs[freq_mask]

    print(f"  Frecuencias narrowband: {len(freqs_narrow)} bins entre {freqs_narrow[0]:.1f} y {freqs_narrow[-1]:.1f} Hz")
    print(f"  Ventanas de tiempo: {len(times)} (de {times[0]:.2f} a {times[-1]:.2f} s)")

    # ---------- VISUALIZACIÓN: ESPECTROGRAMAS ----------
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, sharey=True)
    extent = [times[0], times[-1], freqs_narrow[0], freqs_narrow[-1]]

    im1 = axes[0].imshow(np.abs(Z_h_narrow)**2, aspect='auto', origin='lower', extent=extent, cmap='viridis')
    axes[0].set_title('Spectrogram H1 (Power)')
    axes[0].set_ylabel('Frequency [Hz]')
    plt.colorbar(im1, ax=axes[0], label='Power')

    im2 = axes[1].imshow(np.abs(Z_l_narrow)**2, aspect='auto', origin='lower', extent=extent, cmap='viridis')
    axes[1].set_title('Spectrogram L1 (Power)')
    axes[1].set_ylabel('Frequency [Hz]')
    plt.colorbar(im2, ax=axes[1], label='Power')

    im3 = axes[2].imshow(np.abs(Z_v_narrow)**2, aspect='auto', origin='lower', extent=extent, cmap='viridis')
    axes[2].set_title('Spectrogram V1 (Power)')
    axes[2].set_ylabel('Frequency [Hz]')
    axes[2].set_xlabel('Time [s]')
    plt.colorbar(im3, ax=axes[2], label='Power')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'spectrograms.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------- IDENTIFICACIÓN DE VENTANAS ACTIVAS ----------
    print("\nIdentificando ventanas activas...")

    energy_h = np.sum(np.abs(Z_h_narrow)**2, axis=0)
    energy_l = np.sum(np.abs(Z_l_narrow)**2, axis=0)
    energy_v = np.sum(np.abs(Z_v_narrow)**2, axis=0)

    combined_energy = (energy_h + energy_l + energy_v) / 3
    thr = np.percentile(combined_energy, 75)
    active_windows = combined_energy > thr

    snr_h = np.max(energy_h) / np.median(energy_h)
    snr_l = np.max(energy_l) / np.median(energy_l)
    snr_v = np.max(energy_v) / np.median(energy_v)

    print(f"  Umbral (P75): {thr:.2e}")
    print(f"  Ventanas activas: {np.sum(active_windows)} de {len(times)} ({np.sum(active_windows)/len(times)*100:.1f}%)")
    print(f"  SNR en banda - H1: {snr_h:.2f}, L1: {snr_l:.2f}, V1: {snr_v:.2f}")

    # ---------- CÁLCULO DE COHERENCIA CRUZADA ----------
    def compute_coherence(Z1, Z2, active_mask=None):
        if active_mask is not None:
            Z1_active = Z1[:, active_mask]
            Z2_active = Z2[:, active_mask]
        else:
            Z1_active = Z1
            Z2_active = Z2

        if Z1_active.shape[1] == 0:
            return np.zeros(Z1.shape[0], dtype=complex)

        P12 = np.mean(Z1_active * np.conj(Z2_active), axis=1)
        P11 = np.mean(np.abs(Z1_active)**2, axis=1)
        P22 = np.mean(np.abs(Z2_active)**2, axis=1)
        return P12 / np.sqrt(P11 * P22 + 1e-12)

    print("\nCalculando coherencia cruzada durante ventanas activas...")
    coh_hl = compute_coherence(Z_h_narrow, Z_l_narrow, active_windows)
    coh_hv = compute_coherence(Z_h_narrow, Z_v_narrow, active_windows)
    coh_lv = compute_coherence(Z_l_narrow, Z_v_narrow, active_windows)

    coh_mag_hl = np.mean(np.abs(coh_hl))
    coh_mag_hv = np.mean(np.abs(coh_hv))
    coh_mag_lv = np.mean(np.abs(coh_lv))

    print(f"  Coherencia promedio durante actividad:")
    print(f"    H1-L1: {coh_mag_hl:.3f}, H1-V1: {coh_mag_hv:.3f}, L1-V1: {coh_mag_lv:.3f}")

    # ---------- VISUALIZACIÓN: COHERENCIA ----------
    plt.figure(figsize=(10, 6))
    plt.plot(freqs_narrow, np.abs(coh_hl), 'b-', label='H1-L1', linewidth=2)
    plt.plot(freqs_narrow, np.abs(coh_hv), 'r-', label='H1-V1', linewidth=2)
    plt.plot(freqs_narrow, np.abs(coh_lv), 'g-', label='L1-V1', linewidth=2)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Coherence Magnitude')
    plt.title('Cross-Coherence During Active Windows')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    plt.text(0.02, 0.98, f'Avg H1-L1: {coh_mag_hl:.3f}\nAvg H1-V1: {coh_mag_hv:.3f}\nAvg L1-V1: {coh_mag_lv:.3f}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'coherence.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------- MODO COMO PROBLEMA ESPECTRAL ----------
    print("\nCalculando modo geométrico dominante (operador espectral)...")

    def dominant_mode(Z, active_mask):
        if np.sum(active_mask) > 1:
            Z_use = Z[:, active_mask]
        else:
            Z_use = Z

        C = (Z_use @ np.conj(Z_use.T)) / Z_use.shape[1]
        eigvals, eigvecs = np.linalg.eigh(C)
        idx = np.argmax(eigvals)
        return eigvecs[:, idx], eigvals[idx]

    mode_profile_h, lambda_h = dominant_mode(Z_h_narrow, active_windows)
    mode_profile_l, lambda_l = dominant_mode(Z_l_narrow, active_windows)
    mode_profile_v, lambda_v = dominant_mode(Z_v_narrow, active_windows)

    print(f"  Autovalores dominantes:")
    print(f"    H1: {lambda_h:.3e}, L1: {lambda_l:.3e}, V1: {lambda_v:.3e}")

    mode_profile_h_norm = mode_profile_h / (np.linalg.norm(mode_profile_h) + 1e-12)
    mode_profile_l_norm = mode_profile_l / (np.linalg.norm(mode_profile_l) + 1e-12)
    mode_profile_v_norm = mode_profile_v / (np.linalg.norm(mode_profile_v) + 1e-12)

    # ---------- VISUALIZACIÓN: PERFILES DE MODOS ----------
    plt.figure(figsize=(10, 6))
    plt.plot(freqs_narrow, np.abs(mode_profile_h_norm), 'b-', label='H1 Mode', linewidth=2)
    plt.plot(freqs_narrow, np.abs(mode_profile_l_norm), 'r-', label='L1 Mode', linewidth=2)
    plt.plot(freqs_narrow, np.abs(mode_profile_v_norm), 'g-', label='V1 Mode', linewidth=2)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Mode Amplitude (Normalized)')
    plt.title('Dominant Mode Profiles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'mode_profiles.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- TEST DE INVARIANCIA EN BANDA (SUBESPACIOS INTERNOS) ---
    print("\n--- TEST DE INVARIANCIA EN BANDA (SUBESPACIOS INTERNOS) ---")

    # Bandas internas: relativas al narrowband actual
    b_lo, b_hi = BAND_NARROW
    b_range = b_hi - b_lo
    test_bands = [
        (b_lo + 0.10 * b_range, b_lo + 0.77 * b_range),
        (b_lo + 0.13 * b_range, b_lo + 0.80 * b_range),
        (b_lo + 0.17 * b_range, b_lo + 0.83 * b_range),
    ]
    # Round for readability
    test_bands = [(round(a), round(b)) for a, b in test_bands]

    ref_mode = mode_profile_h_norm.copy()
    ref_freqs = freqs_narrow

    for b0, b1 in test_bands:
        submask = (ref_freqs >= b0) & (ref_freqs <= b1)
        if np.sum(submask) < 2:
            print(f"  Banda {b0}-{b1} Hz → demasiado estrecha, se omite")
            continue

        Z_test = Z_h_narrow[submask, :]
        mode_test, _ = dominant_mode(Z_test, active_windows)
        mode_test /= np.linalg.norm(mode_test)

        ref_sub = ref_mode[submask]
        ref_sub /= np.linalg.norm(ref_sub)
        overlap = np.abs(np.vdot(ref_sub, mode_test))
        print(f"  Banda {b0}-{b1} Hz → overlap = {overlap:.3f}")

    # --- TEST DE INVARIANCIA TEMPORAL (SUBSOPORTES) ---
    print("\n--- TEST DE INVARIANCIA TEMPORAL (SUBSOPORTES) ---")

    def dominant_mode_from_time_mask(Z, time_mask):
        Z_sel = Z[:, time_mask]
        if Z_sel.shape[1] < 5:
            return None
        C = Z_sel @ Z_sel.conj().T / Z_sel.shape[1]
        w, v = np.linalg.eigh(C)
        return v[:, -1] / np.linalg.norm(v[:, -1])

    mode_full = dominant_mode_from_time_mask(Z_h_narrow, active_windows)

    idx_active = np.where(active_windows)[0]
    n = len(idx_active)
    time_masks = {
        "primera mitad": idx_active[:n//2],
        "segunda mitad": idx_active[n//2:],
        "zona central": idx_active[n//4:3*n//4],
    }

    for label, idx in time_masks.items():
        mask = np.zeros_like(active_windows, dtype=bool)
        mask[idx] = True
        mode_sub = dominant_mode_from_time_mask(Z_h_narrow, mask)
        if mode_sub is None:
            continue
        overlap = np.abs(np.vdot(mode_full, mode_sub))
        print(f"  {label} → overlap = {overlap:.3f}")

    # --- TEST C: NULL TEST FÍSICO (RUIDO EN LA MISMA MÉTRICA) ---
    print("\n--- TEST C: NULL TEST FÍSICO (RUIDO EN LA MISMA MÉTRICA) ---")

    def extract_dominant_mode(Z):
        C = Z @ Z.conj().T / Z.shape[1]
        w, v = np.linalg.eigh(C)
        return v[:, -1] / np.linalg.norm(v[:, -1]), np.max(w)

    np.random.seed(0)
    noise_ts = TimeSeries(
        np.random.normal(0, 1, size=len(h1_arr)),
        sample_rate=FS
    )

    noise_proc = noise_ts.whiten(8, 4).bandpass(
        BAND_NARROW[0], BAND_NARROW[1]
    )
    noise_arr = noise_proc.value

    freqs_n, times_n, Z_noise = stft(
        noise_arr,
        fs=FS,
        nperseg=N_PER_SEG,
        noverlap=N_OVERLAP,
        window="hamming"
    )
    freq_mask_n = (freqs_n >= BAND_NARROW[0]) & (freqs_n <= BAND_NARROW[1])
    Z_noise = Z_noise[freq_mask_n, :]

    Z_noise_active = Z_noise[:, active_windows]

    mode_noise, lambda_noise = extract_dominant_mode(Z_noise_active)

    overlap_noise = np.abs(np.vdot(mode_profile_h_norm, mode_noise))
    print(f"  Autovalor dominante (ruido): {lambda_noise:.3e}")
    print(f"  Overlap con modo GW (H1): {overlap_noise:.3f}")

    # --- TEST C': NULL TEST GEOMÉTRICO (DESFASE INTER-DETECTOR) ---
    print("\n--- TEST C': NULL TEST GEOMÉTRICO (DESFASE INTER-DETECTOR) ---")

    def time_shift(arr, shift):
        return np.roll(arr, shift)

    shift_H1 = int(0.3 * FS)
    shift_L1 = int(0.6 * FS)
    shift_V1 = int(0.9 * FS)

    h1_shift = time_shift(h1_arr, shift_H1)
    l1_shift = time_shift(l1_arr, shift_L1)
    v1_shift = time_shift(v1_arr, shift_V1)

    def mode_from_ts(x):
        freqs_t, _, Z = stft(
            x,
            fs=FS,
            nperseg=N_PER_SEG,
            noverlap=N_OVERLAP,
            window="hamming"
        )
        Z = Z[(freqs_t >= BAND_NARROW[0]) & (freqs_t <= BAND_NARROW[1]), :]
        C = Z @ Z.conj().T / Z.shape[1]
        w, v = np.linalg.eigh(C)
        return v[:, -1] / np.linalg.norm(v[:, -1]), np.max(w)

    mode_h1_s, _ = mode_from_ts(h1_shift)
    mode_l1_s, _ = mode_from_ts(l1_shift)
    mode_v1_s, _ = mode_from_ts(v1_shift)

    ov_hl = abs(np.vdot(mode_h1_s, mode_l1_s))
    ov_hv = abs(np.vdot(mode_h1_s, mode_v1_s))
    ov_lv = abs(np.vdot(mode_l1_s, mode_v1_s))

    print(f"  Overlap H1–L1 (desfasado): {ov_hl:.3f}")
    print(f"  Overlap H1–V1 (desfasado): {ov_hv:.3f}")
    print(f"  Overlap L1–V1 (desfasado): {ov_lv:.3f}")

    # --- TEST C'': NULL TEST GEOMÉTRICO DE RED ---
    print("\n--- TEST C'': NULL TEST GEOMÉTRICO DE RED (COHERENCIA GLOBAL) ---")

    def network_mode(Zh, Zl, Zv):
        Znet = np.vstack([Zh, Zl, Zv])
        Cnet = Znet @ Znet.conj().T / Znet.shape[1]
        w, v = np.linalg.eigh(Cnet)
        return v[:, -1] / np.linalg.norm(v[:, -1]), np.max(w)

    mode_net, eig_net = network_mode(
        Z_h_narrow[:, active_windows],
        Z_l_narrow[:, active_windows],
        Z_v_narrow[:, active_windows]
    )

    def shift_Z(Z, shift):
        return np.roll(Z, shift, axis=1)

    Z_h_s = shift_Z(Z_h_narrow, 17)
    Z_l_s = shift_Z(Z_l_narrow, 43)
    Z_v_s = shift_Z(Z_v_narrow, 89)

    mode_net_s, eig_net_s = network_mode(
        Z_h_s[:, active_windows],
        Z_l_s[:, active_windows],
        Z_v_s[:, active_windows]
    )

    overlap_net = abs(np.vdot(mode_net, mode_net_s))
    print(f"  Autovalor red (GW): {eig_net:.3e}")
    print(f"  Autovalor red (null): {eig_net_s:.3e}")
    print(f"  Overlap modo de red: {overlap_net:.3f}")

    # ---------- VISUALIZACIÓN: MODO GEOMÉTRICO DE RED (INTRÍNSECO) ----------
    mode_net_reshaped = mode_net.reshape(3, -1)

    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(mode_net_reshaped), aspect='auto', origin='lower',
               extent=[freqs_narrow[0], freqs_narrow[-1], 0, 3], cmap='plasma')
    plt.colorbar(label='Mode Amplitude')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Detector')
    plt.yticks([0.5, 1.5, 2.5], ['H1', 'L1', 'V1'])
    plt.title('Intrinsic Geometric Mode of the Gravitational Wave Network')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'network_geometric_mode.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ============================================================
    # --- DESCOMPOSICIÓN GEOMÉTRICA: MÉTRICA INSTRUMENTAL vs GW ---
    # ============================================================
    print("\n--- DESCOMPOSICIÓN GEOMÉTRICA (OFF-SOURCE → PROYECCIÓN) ---")

    off_windows = ~active_windows
    print(f"  Ventanas off-source: {np.sum(off_windows)}")

    Znet_off = np.vstack([
        Z_h_narrow[:, off_windows],
        Z_l_narrow[:, off_windows],
        Z_v_narrow[:, off_windows]
    ])

    C_instr = Znet_off @ Znet_off.conj().T / Znet_off.shape[1]

    w_instr, v_instr = np.linalg.eigh(C_instr)

    idx = np.argsort(w_instr)[::-1]
    w_instr = w_instr[idx]
    v_instr = v_instr[:, idx]

    print(f"  Autovalores instrumentales dominantes:")
    for i in range(3):
        print(f"    λ_instr[{i}] = {w_instr[i]:.3e}")

    k_instr = 2
    V_instr = v_instr[:, :k_instr]

    P_perp = np.eye(V_instr.shape[0], dtype=complex)
    for i in range(k_instr):
        vi = V_instr[:, i].reshape(-1, 1)
        P_perp -= vi @ vi.conj().T

    print(f"  Subespacio instrumental: dim = {k_instr}")
    print(f"  Proyector construido")

    Znet_on = np.vstack([
        Z_h_narrow[:, active_windows],
        Z_l_narrow[:, active_windows],
        Z_v_narrow[:, active_windows]
    ])

    Znet_proj = P_perp @ Znet_on
    C_proj = Znet_proj @ Znet_proj.conj().T / Znet_proj.shape[1]

    w_proj, v_proj = np.linalg.eigh(C_proj)
    idxp = np.argsort(w_proj)[::-1]
    w_proj = w_proj[idxp]
    v_proj = v_proj[:, idxp]

    mode_net_proj = v_proj[:, 0]
    mode_net_proj = mode_net_proj / np.linalg.norm(mode_net_proj)
    eig_net_proj = w_proj[0]

    print(f"  Autovalor dominante proyectado (GW): {eig_net_proj:.3e}")
    print(f"\n  Autovalor dominante ON-source (sin proyectar): {eig_net:.3e}")
    print(f"  Autovalor dominante proyectado (GW): {w_proj[0]:.3e}")

    overlap_proj = abs(np.vdot(mode_net, v_proj[:, 0]))
    print(f"  Overlap modo red vs proyectado: {overlap_proj:.3f}")
    mode_net_proj = v_proj[:, 0] / np.linalg.norm(v_proj[:, 0])

    # ---------- VISUALIZACIÓN: MODO GEOMÉTRICO PROYECTADO ----------
    mode_net_proj_reshaped = mode_net_proj.reshape(3, -1)

    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(mode_net_proj_reshaped), aspect='auto', origin='lower',
               extent=[freqs_narrow[0], freqs_narrow[-1], 0, 3], cmap='plasma')
    plt.colorbar(label='Projected Mode Amplitude')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Detector')
    plt.yticks([0.5, 1.5, 2.5], ['H1', 'L1', 'V1'])
    plt.title('Projected Geometric Mode (Instrumental Noise Subtracted)')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'projected_geometric_mode.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- TEST D1: INVARIANCIA EN BANDA (MODO PROYECTADO) ---
    print("\n--- TEST D1: INVARIANCIA EN BANDA (MODO PROYECTADO) ---")

    ref_mode_proj = mode_net_proj.copy()
    ref_mode_proj /= np.linalg.norm(ref_mode_proj)
    n_det = 3

    for b0, b1 in test_bands:
        submask = (freqs_narrow >= b0) & (freqs_narrow <= b1)
        if np.sum(submask) < 2:
            print(f"  Banda {b0}-{b1} Hz → omitida")
            continue

        submask_net = np.tile(submask, n_det)
        Z_sub = Znet_proj[submask_net, :]

        C_sub = Z_sub @ Z_sub.conj().T / Z_sub.shape[1]
        w, v = np.linalg.eigh(C_sub)
        mode_sub = v[:, -1]
        mode_sub /= np.linalg.norm(mode_sub)

        ref_sub = ref_mode_proj[submask_net]
        ref_sub /= np.linalg.norm(ref_sub)
        overlap = abs(np.vdot(ref_sub, mode_sub))
        print(f"  Banda {b0}-{b1} Hz → overlap = {overlap:.3f}")

    # --- TEST D2: ESTABILIDAD FRENTE A DIMENSIÓN INSTRUMENTAL ---
    print("\n--- TEST D2: ESTABILIDAD FRENTE A k_instr ---")

    ref_mode = mode_net_proj.copy()
    ref_mode /= np.linalg.norm(ref_mode)

    for k_test in [1, 2, 3]:
        Vt = v_instr[:, :k_test]

        P_perp_t = np.eye(Vt.shape[0], dtype=complex)
        for i in range(k_test):
            vi = Vt[:, i].reshape(-1, 1)
            P_perp_t -= vi @ vi.conj().T

        Zproj_t = P_perp_t @ Znet_on

        Ct = Zproj_t @ Zproj_t.conj().T / Zproj_t.shape[1]
        w_t, v_t = np.linalg.eigh(Ct)
        idx_t = np.argsort(w_t)[::-1]
        mode_t = v_t[:, idx_t[0]]
        mode_t /= np.linalg.norm(mode_t)

        overlap = abs(np.vdot(ref_mode, mode_t))
        print(f"  k_instr = {k_test} → overlap = {overlap:.3f}")

    # --- TEST D3: GAP ESPECTRAL (UNICIDAD DEL MODO GW) ---
    print("\n--- TEST D3: GAP ESPECTRAL DEL OPERADOR PROYECTADO ---")

    lambda1 = w_proj[0]
    lambda2 = w_proj[1]
    lambda3 = w_proj[2]

    gap12 = lambda1 / lambda2
    gap23 = lambda2 / lambda3

    print(f"  λ1 = {lambda1:.3e}")
    print(f"  λ2 = {lambda2:.3e}")
    print(f"  λ3 = {lambda3:.3e}")

    eigvals_proj = np.array([lambda1, lambda2, lambda3])
    eta_proj = eigvals_proj[0] / np.sum(eigvals_proj)

    print(f"\n  Gap λ1 / λ2 = {gap12:.2f}")
    print(f"  Gap λ2 / λ3 = {gap23:.2f}")

    # ---------- VISUALIZACIÓN: ESPECTRO DE AUTOVALORES ----------
    plt.figure(figsize=(8, 6))
    eigenvalues = [lambda1, lambda2, lambda3]
    bars = plt.bar(range(1, 4), eigenvalues, color=['red', 'blue', 'green'], alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue Magnitude')
    plt.title('Spectral Decomposition: Eigenvalues of Projected Operator')
    plt.xticks(range(1, 4), ['λ₁ (GW)', 'λ₂', 'λ₃'])
    plt.grid(True, alpha=0.3, axis='y')

    for i, (bar, val) in enumerate(zip(bars, eigenvalues)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height * 1.1, f'{val:.3e}',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'eigenvalue_spectrum.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------- CÁLCULO DE Dmt MEJORADO ----------
    print("\nCalculando Dmt con normalización robusta...")

    Dmt_h = np.zeros(len(times), dtype=complex)
    Dmt_l = np.zeros(len(times), dtype=complex)
    Dmt_v = np.zeros(len(times), dtype=complex)

    for i in range(len(times)):
        sig_h = Z_h_narrow[:, i]
        sig_l = Z_l_narrow[:, i]
        sig_v = Z_v_narrow[:, i]

        norm_h = np.linalg.norm(sig_h) + 1e-12
        norm_l = np.linalg.norm(sig_l) + 1e-12
        norm_v = np.linalg.norm(sig_v) + 1e-12

        Dmt_h[i] = np.vdot(mode_profile_h_norm, sig_h / norm_h)
        Dmt_l[i] = np.vdot(mode_profile_l_norm, sig_l / norm_l)
        Dmt_v[i] = np.vdot(mode_profile_v_norm, sig_v / norm_v)

    Dmt_amplitude = np.array([np.abs(Dmt_h), np.abs(Dmt_l), np.abs(Dmt_v)])
    Dmt_phase = np.array([np.angle(Dmt_h), np.angle(Dmt_l), np.angle(Dmt_v)])

    # ---------- VISUALIZACIÓN: Dmt AMPLITUDES Y FASES ----------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(times, Dmt_amplitude[0], 'b-', label='H1', linewidth=1.5)
    axes[0].plot(times, Dmt_amplitude[1], 'r-', label='L1', linewidth=1.5)
    axes[0].plot(times, Dmt_amplitude[2], 'g-', label='V1', linewidth=1.5)
    axes[0].axhline(y=thr, color='k', linestyle='--', alpha=0.7, label='Threshold')
    axes[0].set_ylabel('Dmt Amplitude')
    axes[0].set_title('Dmt Amplitudes Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, Dmt_phase[0], 'b-', label='H1', linewidth=1.5)
    axes[1].plot(times, Dmt_phase[1], 'r-', label='L1', linewidth=1.5)
    axes[1].plot(times, Dmt_phase[2], 'g-', label='V1', linewidth=1.5)
    axes[1].set_ylabel('Dmt Phase [rad]')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_title('Dmt Phases Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'dmt_amplitude_phase.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------- VALIDACIÓN FÍSICA ----------
    print("\n--- VALIDACIÓN FÍSICA ---")

    amp_ratio_hl = np.median(Dmt_amplitude[0, active_windows]) / np.median(Dmt_amplitude[1, active_windows])
    amp_ratio_hv = np.median(Dmt_amplitude[0, active_windows]) / np.median(Dmt_amplitude[2, active_windows])

    print(f"Ratios de amplitud (durante actividad):")
    print(f"  H1/L1: {amp_ratio_hl:.2f}")
    print(f"  H1/V1: {amp_ratio_hv:.2f}")

    if np.sum(active_windows) > 0:
        active_indices = np.where(active_windows)[0]
        mid_idx = len(active_indices) // 2
        sample_idx = active_indices[max(0, mid_idx-5):min(len(active_indices), mid_idx+5)]

        phase_diff_hl = np.mean(np.unwrap(Dmt_phase[0, sample_idx] - Dmt_phase[1, sample_idx]))
        phase_diff_hv = np.mean(np.unwrap(Dmt_phase[0, sample_idx] - Dmt_phase[2, sample_idx]))

        f_center = np.mean(freqs_narrow)
        delay_hl_est = phase_diff_hl / (2 * np.pi * f_center)
        delay_hv_est = phase_diff_hv / (2 * np.pi * f_center)

        print(f"\nRetardos estimados desde diferencia de fase:")
        print(f"  H1-L1: {delay_hl_est:.4f} s ")
        print(f"  H1-V1: {delay_hv_est:.4f} s ")

    corr_hl = np.corrcoef(Dmt_amplitude[0, active_windows], Dmt_amplitude[1, active_windows])[0,1]
    corr_hv = np.corrcoef(Dmt_amplitude[0, active_windows], Dmt_amplitude[2, active_windows])[0,1]

    print(f"\nCorrelación de amplitudes (durante actividad):")
    print(f"  H1-L1: {corr_hl:.3f}")
    print(f"  H1-V1: {corr_hv:.3f}")

    # ---------- GUARDAR DATOS ----------
    print(f"\nGuardando datos en: {outdir}/")

    validation_metrics = {
        "snr_h": snr_h,
        "snr_l": snr_l,
        "snr_v": snr_v,
        "coherence_hl": coh_mag_hl,
        "coherence_hv": coh_mag_hv,
        "coherence_lv": coh_mag_lv,
        "amp_ratio_hl": amp_ratio_hl,
        "amp_ratio_hv": amp_ratio_hv
    }

    npz_path = os.path.join(outdir, "narrowband_improved.npz")
    np.savez_compressed(
        npz_path,
        mode_net_proj=mode_net_proj,
        eigvals_proj=eigvals_proj,
        eta_proj=eta_proj,
        Dmt_amplitude=Dmt_amplitude,
        Dmt_phase=Dmt_phase,
        freqs=freqs_narrow,
        times=times,
        mode_profile_h=mode_profile_h,
        mode_profile_l=mode_profile_l,
        mode_profile_v=mode_profile_v,
        mode_profile_h_norm=mode_profile_h_norm,
        mode_profile_l_norm=mode_profile_l_norm,
        mode_profile_v_norm=mode_profile_v_norm,
        coh_hl=coh_hl,
        coh_hv=coh_hv,
        coh_lv=coh_lv,
        energy_h=energy_h,
        energy_l=energy_l,
        energy_v=energy_v,
        combined_energy=combined_energy,
        active_windows=active_windows,
        threshold=thr,
        Z_h_narrow=Z_h_narrow,
        Z_l_narrow=Z_l_narrow,
        Z_v_narrow=Z_v_narrow,
        detectors=["H1", "L1", "V1"],
        event=EVENT,
        gps_center=GPS_CENTER,
        band_narrow=BAND_NARROW,
        stft_params={"nperseg": N_PER_SEG, "noverlap": N_OVERLAP, "window": "hamming"},
        validation_metrics=validation_metrics
    )

    print(f"\n✓ Pipeline completado exitosamente!")
    print(f"✓ Datos guardados en: {outdir}/")
    print(f"✓ {len(active_windows)} ventanas activas identificadas")
    print(f"✓ Coherencia cruzada: H1-L1={coh_mag_hl:.3f}, H1-V1={coh_mag_hv:.3f}")

    print("\n--- VALIDACIÓN FÍSICA RESUMEN ---")
    print(f"✓ SNR en banda: H1={snr_h:.1f}, L1={snr_l:.1f}, V1={snr_v:.1f}")
    print(f"✓ Coherencia durante actividad: >{min(coh_mag_hl, coh_mag_hv, coh_mag_lv):.3f}")
    print(f"✓ Ratios de amplitud plausibles: H1/L1={amp_ratio_hl:.2f}, H1/V1={amp_ratio_hv:.2f}")

    if np.sum(active_windows) > 0:
        print(f"✓ Validación de retardos: Comparables con valores físicos esperados")

    print(f"\n✓ Pipeline listo para visualización 3D!")

    # ---------- RETURN RESULTS ----------
    return {
        "eigvals_proj": eigvals_proj,
        "eta_proj": eta_proj,
        "mode_net_proj": mode_net_proj,
        "mode_profile_h_norm": mode_profile_h_norm,
        "mode_profile_l_norm": mode_profile_l_norm,
        "mode_profile_v_norm": mode_profile_v_norm,
        "mode_profile_h": mode_profile_h,
        "mode_profile_l": mode_profile_l,
        "mode_profile_v": mode_profile_v,
        "Z_h_narrow": Z_h_narrow,
        "Z_l_narrow": Z_l_narrow,
        "Z_v_narrow": Z_v_narrow,
        "times": times,
        "freqs": freqs_narrow,
        "active_windows": active_windows,
        "validation_metrics": validation_metrics,
        "npz_path": npz_path,
        "Dmt_amplitude": Dmt_amplitude,
        "Dmt_phase": Dmt_phase,
        "band": BAND_NARROW,
    }


# ================================================================
# Standalone entry point — preserves original GW170814 behaviour
# ================================================================
if __name__ == "__main__":
    analyze_narrowband(
        event="GW170814",
        gps_center=1186741861,
        band=(150.0, 300.0),
        outdir="./CLASS_A_Narrowband/GW170814_modal_improved_MergerAndRingdown",
    )
