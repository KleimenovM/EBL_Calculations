import numpy as np
from src.source_base import Source


def log_likelihood_single_source(source: Source, model: np.ndarray):
    """
    Calculates the log-likelihood for a single source given a model and observed data.

    :param source: [Source] An instance of the `Source` class containing observed `dnde`
        data, upper errors (`dnde_errp`), and lower errors (`dnde_errn`).
    :param model: [numpy.array] the predicted `dnde` values of the model corresponding to the observation points
    :return: [float]  the negative total log-likelihood based on the observed data and model predictions
    """
    delta = model - source.dnde
    log_likelihood_plus = np.heaviside(delta, 0.5) * 0.5 * (delta / source.dnde_errp) ** 2
    log_likelihood_minus = np.heaviside(-delta, 0.5) * 0.5 * (delta / source.dnde_errn) ** 2
    return - np.sum(log_likelihood_plus + log_likelihood_minus)


def log_norm(x, mu, sigma):
    """
    Calculate the log pdf of a normal distribution
    :param x: value
    :param mu: center of a normal distribution
    :param sigma: variance of a normal distribution
    :return: parabolic_function - const(sigma)
    """
    return -(x - mu) ** 2 / (2 * sigma ** 2)


def full_log_norm(x, mu, sigma):
    return -(x - mu) ** 2 / (2 * sigma ** 2) - 0.5 * np.log(2 * np.pi * sigma ** 2)


def log_uniform(x, center, half_width):
    """
    Calculate the log pdf of a uniform distribution
    :param x: value
    :param center: center of a uniform distribution
    :param half_width: half-width of a uniform distribution
    :return: log(1/2hw) if x is in the uniform distribution, -np.inf otherwise
    """
    return 0 if abs(x - center) < half_width else - np.inf


def full_log_uniform(x, center, half_width):
    return np.log(1 / (2 * half_width)) if abs(x - center) < half_width else -np.inf
