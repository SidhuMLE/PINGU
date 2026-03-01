"""Confidence ellipses and uncertainty quantification for position estimates."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2 as chi2_dist

from pingu.types import PositionEstimate


def confidence_ellipse(
    covariance_2x2: NDArray[np.float64],
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute the confidence ellipse parameters from a 2x2 covariance matrix.

    The ellipse encloses the given confidence level of a 2-D Gaussian
    distribution with the specified covariance.

    Args:
        covariance_2x2: 2x2 position covariance matrix (m^2).
        confidence: Confidence level in (0, 1). Default 0.95.

    Returns:
        Tuple of (semi_major, semi_minor, angle_degrees) where:
            - semi_major: Semi-major axis length in meters.
            - semi_minor: Semi-minor axis length in meters.
            - angle_degrees: Rotation angle of the major axis from the x-axis,
              measured counter-clockwise, in degrees.
    """
    cov = np.asarray(covariance_2x2, dtype=np.float64)
    if cov.shape != (2, 2):
        raise ValueError(f"Expected 2x2 covariance matrix, got shape {cov.shape}.")

    # Chi-squared critical value for 2 degrees of freedom
    chi2_val = chi2_dist.ppf(confidence, df=2)

    # Eigendecomposition (eigenvalues returned in ascending order)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Clamp negative eigenvalues (numerical noise) to zero
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Semi-axes: sqrt(chi2 * lambda)
    semi_minor = float(np.sqrt(chi2_val * eigenvalues[0]))
    semi_major = float(np.sqrt(chi2_val * eigenvalues[1]))

    # Angle of the major axis (eigenvector corresponding to largest eigenvalue)
    major_eigvec = eigenvectors[:, 1]
    angle_rad = np.arctan2(major_eigvec[1], major_eigvec[0])
    angle_degrees = float(np.degrees(angle_rad))

    return (semi_major, semi_minor, angle_degrees)


def confidence_radius(
    covariance_2x2: NDArray[np.float64],
    confidence: float = 0.95,
) -> float:
    """Compute a circular confidence radius from a 2x2 covariance matrix.

    This is a conservative circular approximation: the radius of a circle
    that would contain the specified confidence level, using the largest
    eigenvalue of the covariance as the variance in all directions.

    Args:
        covariance_2x2: 2x2 position covariance matrix (m^2).
        confidence: Confidence level in (0, 1). Default 0.95.

    Returns:
        Confidence radius in meters.
    """
    cov = np.asarray(covariance_2x2, dtype=np.float64)
    if cov.shape != (2, 2):
        raise ValueError(f"Expected 2x2 covariance matrix, got shape {cov.shape}.")

    chi2_val = chi2_dist.ppf(confidence, df=2)
    eigenvalues = np.linalg.eigvalsh(cov)
    max_eigenvalue = float(np.maximum(eigenvalues[-1], 0.0))
    return float(np.sqrt(chi2_val * max_eigenvalue))


def position_uncertainty(
    position_estimate: PositionEstimate,
    confidence: float = 0.95,
) -> dict:
    """Summarize position uncertainty from a PositionEstimate.

    Args:
        position_estimate: Solved position with covariance.
        confidence: Confidence level. Default 0.95.

    Returns:
        Dict with keys:
            - semi_major: Semi-major axis of the confidence ellipse (m).
            - semi_minor: Semi-minor axis of the confidence ellipse (m).
            - angle_degrees: Orientation of the major axis (degrees from x-axis).
            - radius: Conservative circular confidence radius (m).
            - cep: Circular Error Probable, i.e., radius enclosing 50%
              of the probability (m).
    """
    cov = position_estimate.covariance

    # Full confidence ellipse
    semi_major, semi_minor, angle_deg = confidence_ellipse(cov, confidence)

    # Conservative circular radius
    radius = confidence_radius(cov, confidence)

    # CEP: Circular Error Probable (50% confidence radius)
    cep = confidence_radius(cov, confidence=0.50)

    return {
        "semi_major": semi_major,
        "semi_minor": semi_minor,
        "angle_degrees": angle_deg,
        "radius": radius,
        "cep": cep,
    }
