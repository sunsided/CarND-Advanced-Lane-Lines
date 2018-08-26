import numpy as np


def curvature_radius(fit, y: float, mx: float) -> float:
    coeff_a, coeff_b, coeff_c = fit
    radius = ((1. + (2. * coeff_a * y + coeff_b) ** 2.) ** 1.5) / np.abs(2. * coeff_a)
    return radius * mx
