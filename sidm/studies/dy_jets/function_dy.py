import awkward as ak
import coffea.util
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import yaml

def normalize_hist(hist):
    """Normalize a histogram to unit area, accounting for bin widths."""
    values = hist.values()
    variances = hist.variances()
    bin_widths = np.diff(hist.axes[0].edges)
    norm = np.sum(values * bin_widths)
    if norm == 0:
        raise ValueError("Cannot normalize a histogram with zero integral.")
    hist.values()[...] = values / norm
    hist.variances()[...] = variances / norm**2
    return hist

def integrate_hist(hist):
    """Integrate a histogram, accounting for bin widths."""
    values = hist.values()
    variances = hist.variances()
    bin_widths = np.diff(hist.axes[0].edges)

    integral = np.sum(values * bin_widths)
    integral_err = np.sqrt(np.sum(variances * bin_widths**2))

    return integral, integral_err

