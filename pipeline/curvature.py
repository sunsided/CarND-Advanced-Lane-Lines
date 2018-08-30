import numpy as np

CURVATURE_INVALID = float('inf')


def curvature_valid(x) -> bool:
    return not np.isinf(x)


def curvature_radius(fit, y: float, mx: float) -> float:
    coeff_a, coeff_b, coeff_c = fit
    if coeff_a == 0:
        return CURVATURE_INVALID
    radius = ((1. + (2. * coeff_a * y + coeff_b) ** 2.) ** 1.5) / (2. * coeff_a)
    return radius * mx
