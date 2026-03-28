"""Bias correction module for the ABC-HP system."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Union

import numpy as np


LOGGER = logging.getLogger(__name__)
NumberLike = Union[float, int, np.ndarray]


@dataclass
class BiasWeights:
    """Configurable coefficients for expected risk calculation."""

    alpha: float = 0.25
    beta: float = 0.25
    gamma: float = 0.25
    delta: float = 0.25


class BiasCorrection:
    """Computes expected risk, bias factor, and corrected risk."""

    def __init__(self, weights: BiasWeights | None = None, epsilon: float = 1e-6) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be greater than 0")

        self.weights = weights or BiasWeights()
        self.epsilon = epsilon

    @staticmethod
    def _to_array(values: NumberLike) -> np.ndarray:
        """Convert scalar/array-like inputs into numpy arrays for vectorized math."""
        return np.asarray(values, dtype=float)

    @staticmethod
    def _to_output(values: np.ndarray) -> NumberLike:
        """Return python float for scalar results, otherwise return ndarray."""
        if values.ndim == 0:
            return float(values)
        return values

    def compute_expected_risk(self, T: NumberLike, W: NumberLike, P: NumberLike, I: NumberLike) -> NumberLike:
        """Compute expected risk: Re = alpha*T + beta*W + gamma*P + delta*I."""
        t = self._to_array(T)
        w = self._to_array(W)
        p = self._to_array(P)
        i = self._to_array(I)

        expected_risk = (
            self.weights.alpha * t
            + self.weights.beta * w
            + self.weights.gamma * p
            + self.weights.delta * i
        )

        LOGGER.debug("Computed expected risk")
        return self._to_output(expected_risk)

    def compute_bias_factor(self, Re: NumberLike, Ro: NumberLike) -> NumberLike:
        """Compute bias factor with zero-safe denominator: B = Re / (Ro + epsilon)."""
        expected = self._to_array(Re)
        observed = self._to_array(Ro)

        # Apply epsilon to avoid divide-by-zero and unstable tiny denominators.
        safe_denominator = np.where(np.abs(observed) < self.epsilon, self.epsilon, observed)
        bias_factor = expected / safe_denominator

        LOGGER.debug("Computed bias factor")
        return self._to_output(bias_factor)

    def compute_corrected_risk(self, Ro: NumberLike, B: NumberLike) -> NumberLike:
        """Compute corrected risk: Rc = Ro * B."""
        observed = self._to_array(Ro)
        bias_factor = self._to_array(B)

        corrected_risk = observed * bias_factor

        LOGGER.debug("Computed corrected risk")
        return self._to_output(corrected_risk)
