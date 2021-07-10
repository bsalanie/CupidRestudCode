""" a variety of functions with their derivatives """

from math import log, exp
from typing import Optional


def bslog(x: float, eps: Optional[float] = 1e-30) -> float:
    """
    extends the logarithm below `eps` by taking a second-order approximation

    :param float x: argument

    :param float eps: lower bound

    :return: :math:`\\ln(x)` :math:`C^2`-extended below `eps`
    """
    if x > eps:
        return log(x)
    else:
        return log(eps) - (eps - x) * (3.0 * eps - x) / (2.0 * eps * eps)


def der_bslog(x: float, eps: Optional[float] = 1e-30) -> float:
    """
    derivative of second-order approximation of logarithm below `eps`

    :param float x: argument

    :param float eps: lower bound

    :return: derivative of :math:`\\ln(x)` :math:`C^2`-extended below `eps`
    """
    if x > eps:
        return 1.0 / x
    else:
        return (2.0 * eps - x) / (eps * eps)


def der2_bslog(x: float, eps: Optional[float] = 1e-30) -> float:
    """
    second derivative of second-order approximation of logarithm below `eps`

    :param float x: argument

    :param float eps: lower bound

    :return: second derivative of :math:`\\ln(x)` :math:`C^2`-extended below `eps`
    """
    if x > eps:
        return -1.0 / (x * x)
    else:
        return -1.0 / (eps * eps)


def bsxlogx(x: float, eps: Optional[float] = 1e-30) -> float:
    """
    extends :math:`x \\ln(x)`  below `eps` by making it go to zero in a :math:`C^1` extension

    :param float x: argument

    :param float eps: lower bound

    :return: :math:`x \\ln(x)`  :math:`C^1`-extended below `eps`
    """
    if x > eps:
        return x * log(x)
    else:
        return eps * log(eps) * (x / eps)


def der_bsxlogx(x: float, eps: Optional[float] = 1e-30) -> float:
    """
    derivative of :math:`C^1` extension of :math:`x \\ln(x)` below `eps`

    :param float x: argument

    :param float eps: lower bound

    :return: derivative of :math:`x \\ln(x)`  :math:`C^1`-extended below `eps`
    """
    if x > eps:
        return 1.0 + log(x)
    else:
        return 2.0 * x / eps + log(eps) - 1.0


def der2_bsxlogx(x: float, eps: Optional[float] = 1e-30) -> float:
    """
    second  derivative of :math:`C^1` extension of :math:`x \\ln(x)` below `eps`

    :param float x: argument

    :param float eps: lower bound

    :return: second derivative of :math:`x \\ln(x)`  :math:`C^1`-extended below `eps`
    """
    if x > eps:
        return 1.0 / x
    else:
        return 2.0 / eps


def bsexp(x: float, bigx: Optional[float] = 30.0, lowx: Optional[float] = -100.0) -> float:
    """
    :math:`C^2`-extends the exponential above `bigx` and below `lowx`

    :param float x: argument

    :param float bigx: upper bound

    :param float lowx: lower bound

    :return: exponential :math:`C^2`-extended above `bigx` and below `lowx`
    """
    if lowx < x < bigx:
        return exp(x)
    elif x < lowx:
        elowx = exp(lowx)
        dx = lowx - x
        exp_smaller = elowx / (1.0 + dx * (1.0 + 0.5 * dx))
        return exp_smaller
    else:
        ebigx = exp(bigx)
        dx = x - bigx
        exp_larger = ebigx * (1.0 + dx * (1.0 + 0.5 * dx))
        return exp_larger


def der_bsexp(x: float, bigx: Optional[float] = 30.0, lowx: Optional[float] = -100.0) -> float:
    """
    derivative of  :math:`C^2`-extended exponential above `bigx`  and below `lowx`

    :param float x: argument

    :param float bigx: upper bound

    :param float lowx: lower bound

    :return: derivative of exponential :math:`C^2`-extended above `bigx` and below `lowx`
    """
    if lowx < x < bigx:
        return exp(x)
    elif x < lowx:
        elowx = exp(lowx)
        dx = lowx - x
        denom = 1.0 + dx * (1.0 + 0.5 * dx)
        exp_smaller = elowx * (1.0 + dx) / (denom * denom)
        return exp_smaller
    else:
        ebigx = exp(bigx)
        dx = x - bigx
        exp_larger = ebigx * (1.0 + dx)
        return exp_larger

