# classify.py
# Geometric event classification based on projected network spectral operator.
# Logic extracted from RT/nrik.py (multievent_analysis).

import numpy as np

REG_EPS = 1e-12


def classify_event(results):
    """
    Classify a gravitational-wave event based on geometric coherence metrics.

    Parameters
    ----------
    results : dict
        Output of analyze_narrowband() for one or more phases.
        Expected keys: 'eigvals_proj', 'eta_proj'.
        If multiple phases are present (e.g. 'inspiral', 'merger_ringdown'),
        classification uses the merger_ringdown band (highest SNR).

    Returns
    -------
    dict
        Classification result with keys:
        - event: str (if present in results)
        - eta: float, coherent energy fraction
        - lambda1_lambda2: float, eigenvalue ratio
        - lambda2_lambda3: float
        - class: str, one of "A", "B", "B+", "C"
    """
    # If results dict contains phase-level results, pick merger_ringdown
    if 'merger_ringdown' in results:
        r = results['merger_ringdown']
    elif 'eigvals_proj' in results:
        r = results
    else:
        # Try to find any phase with eigvals_proj
        for key in ['merger_ringdown', 'post_inspiral', 'inspiral']:
            if key in results and 'eigvals_proj' in results[key]:
                r = results[key]
                break
        else:
            raise ValueError("No eigvals_proj found in results")

    eigvals = np.array(r["eigvals_proj"], dtype=float)
    eigvals = np.sort(eigvals)[::-1]
    eta = float(r["eta_proj"])

    r12 = eigvals[0] / max(eigvals[1], REG_EPS)
    r23 = eigvals[1] / max(eigvals[2], REG_EPS)

    if eta > 0.35 and r12 > 1.3 and r23 > 1.3:
        geom_class = "A"
    elif eta > 0.25 and r12 > 1.1 and r23 > 1.3:
        geom_class = "B"
    elif eta > 0.20:
        geom_class = "B+"
    else:
        geom_class = "C"

    return {
        "eta": eta,
        "lambda1_lambda2": r12,
        "lambda2_lambda3": r23,
        "eigvals": eigvals,
        "class": geom_class,
    }


def classify_events_batch(results_list):
    """
    Classify multiple events and produce a summary table.

    Parameters
    ----------
    results_list : list of dict
        Each element: { "event": str, "results": phase-results dict }

    Returns
    -------
    list of dict
        Classification for each event.
    """
    summary = []
    for entry in results_list:
        event = entry.get("event", "unknown")
        try:
            cls = classify_event(entry["results"])
            cls["event"] = event
            summary.append(cls)
        except Exception as e:
            print(f"⚠️  Could not classify {event}: {e}")

    return summary


def print_classification(classification):
    """Pretty-print a single event classification."""
    print(f"  η = {classification['eta']:.3f}")
    print(f"  λ1/λ2 = {classification['lambda1_lambda2']:.2f}")
    print(f"  λ2/λ3 = {classification['lambda2_lambda3']:.2f}")
    print(f"  → Class {classification['class']}")


def print_summary_table(summary):
    """Print classification summary for multiple events."""
    print(f"\n{'Event':<12s} | {'η':>6s} | {'λ1/λ2':>6s} | {'λ2/λ3':>6s} | Class")
    print("-" * 55)
    for r in summary:
        print(
            f"{r['event']:<12s} | "
            f"{r['eta']:.3f} | "
            f"{r['lambda1_lambda2']:.2f} | "
            f"{r['lambda2_lambda3']:.2f} | "
            f"{r['class']}"
        )
