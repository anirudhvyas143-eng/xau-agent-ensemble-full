# drift_manager.py — simple change-of-distribution detection stub
import numpy as np

def psi(expected, actual, buckets=10):
    """
    Population Stability Index (PSI) between two arrays.
    small values ~ stable.
    """
    expected = np.array(expected).flatten()
    actual = np.array(actual).flatten()
    if len(expected)==0 or len(actual)==0:
        return 0.0
    e_perc, _ = np.histogram(expected, bins=buckets, density=True)
    a_perc, _ = np.histogram(actual, bins=buckets, density=True)
    e_perc = np.where(e_perc==0, 1e-8, e_perc)
    a_perc = np.where(a_perc==0, 1e-8, a_perc)
    psi_vals = (e_perc - a_perc) * np.log(e_perc / a_perc)
    return float(np.sum(psi_vals))
