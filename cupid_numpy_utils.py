""" useful functions that rely on Numpy """

import numpy as np
from typing import Union, Optional, Tuple
from math import log, exp
import sys

from cupid_utils import bs_error_abort, print_stars, bs_name_func


def _report_small_args(arr: np.ndarray, eps: float, func_name: str) -> None:
    """
    reports on small arguments

    :param arr: any Numpy array

    :param eps: a positive number

    :param func_name: the name of the calling function

    :return: nothing
    """
    n_small_args = np.sum(arr < eps)
    if n_small_args > 0:
        finals = 's' if n_small_args > 1 else ''
        print(f"{func_name}: {n_small_args} argument{finals} smaller than {eps}: mini = {np.min(arr)}")
    return


def _report_large_args(arr: np.ndarray, bigx: float, func_name: str) -> None:
    """
    reports on large arguments

    :param arr: any Numpy array

    :param bigx: a positive number

    :param func_name: the name of the calling function

    :return: nothing
    """
    n_large_args = np.sum(arr > bigx)
    if n_large_args > 0:
        finals = 's' if n_large_args  > 1 else ''
        print(f"{func_name}: {n_large_args } argument{finals} larger than {bigx}: maxi = {np.max(arr)}")
    return


def nplog(arr: np.ndarray, eps: Optional[float] = 1e-30, verbose: Optional[bool] = False) -> np.ndarray:
    """
    :math:`C^2` extension of  :math:`\\ln(a)` below `eps`

    :param arr: a Numpy array

    :param eps: lower bound

    :param verbose: if True, reports diagnosis

    :return:  :math:`\\ln(a)` :math:`C^2`-extended below `eps`
    """
    if np.min(arr) > eps:
        return np.log(arr)
    else:
        logarreps = np.log(np.maximum(arr, eps))
        logarr_smaller = log(eps) - (eps - arr) * (3.0 * eps - arr) / (2.0 * eps * eps)
        if verbose:
            _report_small_args(arr, eps, bs_name_func())
        return np.where(arr > eps, logarreps, logarr_smaller)





def der_nplog(arr: np.ndarray, eps: Optional[float] = 1e-30, verbose: Optional[bool] = False) -> np.ndarray:
    """
    derivative of :math:`C^2` extension of  :math:`\\ln(a)` below `eps`

    :param np.ndarray arr: a Numpy array

    :param float eps: lower bound

    :param verbose: if True, reports diagnosis

    :return: derivative of  :math:`\\ln(a)` :math:`C^2`-extended below `eps`
    """
    if np.min(arr) > eps:
        return 1.0 / arr
    else:
        der_logarreps = 1.0 / np.maximum(arr, eps)
        der_logarr_smaller = (2.0 * eps - arr) / (eps * eps)
        if verbose:
            _report_small_args(arr, eps, bs_name_func())
        return np.where(arr > eps, der_logarreps, der_logarr_smaller)


def der2_nplog(arr: np.ndarray, eps: Optional[float] = 1e-30, verbose: Optional[bool] = False) -> np.ndarray:
    """
    second derivative of :math:`C^2` extension of  :math:`\\ln(a)` below `eps`

    :param np.ndarray arr: a Numpy array

    :param float eps: lower bound

    :param verbose: if True, reports diagnosis

    :return:  second derivative of :math:`\\ln(a)` :math:`C^2`-extended below `eps`
    """
    if np.min(arr) > eps:
        return -1.0 / (arr * arr)
    else:
        arreps = np.maximum(arr, eps)
        der2_logarreps = -1.0 / (arreps * arreps)
        der2_logarr_smaller = np.full(arr.shape, -1.0 / (eps * eps))
        if verbose:
            _report_small_args(arr, eps, bs_name_func())
        return np.where(arr > eps, der2_logarreps, der2_logarr_smaller)


def npxlogx(arr: np.ndarray, eps: Optional[float] = 1e-30, verbose: Optional[bool] = False) -> np.ndarray:
    """
    :math:`C^2` extension of  :math:`a\\ln(a)` below `eps`

    :param np.ndarray arr: a Numpy array

    :param float eps: lower bound

    :param verbose: if True, reports diagnosis

    :return:  :math:`a\\ln(a)`  :math:`C^2`-extended  below `eps`
    """
    if np.min(arr) > eps:
        return arr * np.log(arr)
    else:
        xlogarreps = arr * np.log(np.maximum(arr, eps))
        xlogarr_smaller = arr * (arr / eps + log(eps) - 1.0)
        if verbose:
            _report_small_args(arr, eps, bs_name_func())
        return np.where(arr > eps, xlogarreps, xlogarr_smaller)


def npexp(arr: np.ndarray, bigx: Optional[float] = 100.0, lowx: Optional[float] = -100.0,
          verbose: Optional[bool] = False) -> np.ndarray:
    """
    :math:`C^2` extension of  :math:`\\exp(a)` above `bigx` and below `lowx`

    :param np.ndarray arr: a Numpy array

    :param float bigx: upper bound

    :param float lowx: lower bound

    :param verbose: if True, reports diagnosis

    :return:   :math:\\exp(a)`  :math:`C^2`-extended above `bigx` and below `lowx`
    """
    min_arr, max_arr = np.min(arr), np.max(arr)
    if max_arr <= bigx and min_arr >= lowx:
        return np.exp(arr)
    elif max_arr > bigx and min_arr > lowx:  # some large arguments, no small ones
        exparr = np.exp(np.minimum(arr, bigx))
        ebigx = exp(bigx)
        darr = arr - bigx
        exparr_larger = ebigx * (1.0 + darr * (1.0 + 0.5 * darr))
        if verbose:
            _report_large_args(arr, bigx, bs_name_func())
        return np.where(arr < bigx, exparr, exparr_larger)
    elif max_arr < bigx and min_arr < lowx:  # some small arguments, no large ones
        exparr = np.exp(np.maximum(arr, lowx))
        elowx = exp(lowx)
        darr = lowx - arr
        exparr_smaller = elowx / (1.0 + darr * (1.0 + 0.5 * darr))
        if verbose:
            _report_small_args(arr, lowx, bs_name_func())
        return np.where(arr > lowx, exparr, exparr_smaller)
    else:  # some small arguments and some large ones
        exparr = np.exp(np.minimum(np.maximum(arr, lowx), bigx))
        elowx = exp(lowx)
        darr_low = lowx - arr
        ebigx = exp(bigx)
        darr_big = arr - bigx
        exparr_smaller = elowx / (1.0 + darr_low * (1.0 + 0.5 * darr_low))
        exparr_larger = ebigx * (1.0 + darr_big * (1.0 + 0.5 * darr_big))
        if verbose:
            _report_large_args(arr, bigx, bs_name_func())
            _report_small_args(arr, lowx, bs_name_func())
        resus = exparr
        resus[arr < lowx] = exparr_smaller[arr < lowx]
        resus[arr > bigx] = exparr_larger[arr > bigx]
        return resus


def der_npexp(arr: np.ndarray, bigx: Optional[float] = 100.0, lowx: Optional[float] = -100.0,
              verbose: Optional[bool] = False) -> np.ndarray:
    """
    derivative of :math:`C^2` extension of  :math:`\\exp(a)` above `bigx` and below `lowx`

    :param np.ndarray arr: a Numpy array

    :param float bigx: upper bound

    :param float lowx: lower bound

    :param verbose: if True, reports diagnosis

    :return: derivative of :math:\\exp(a)`  :math:`C^2`-extended above `bigx`  and below `lowx`
    """
    min_arr, max_arr = np.min(arr), np.max(arr)
    if max_arr <= bigx and min_arr >= lowx:
        return np.exp(arr)
    elif max_arr > bigx and min_arr > lowx:  # some large arguments, no small ones
        exparr = np.exp(np.minimum(arr, bigx))
        ebigx = exp(bigx)
        darr = arr - bigx
        exparr_larger = ebigx * (1.0 + darr)
        if verbose:
            _report_large_args(arr, bigx, bs_name_func())
        return np.where(arr < bigx, exparr, exparr_larger)
    elif max_arr < bigx and min_arr < lowx:  # some small arguments, no large ones
        exparr = np.exp(np.maximum(arr, lowx))
        elowx = exp(lowx)
        darr = lowx - arr
        denom = 1.0 + darr * (1.0 + 0.5 * darr)
        exparr_smaller = elowx * (1.0 + darr) / (denom * denom)
        if verbose:
            _report_small_args(arr, lowx, bs_name_func())
        return np.where(arr > lowx, exparr, exparr_smaller)
    else:  # some small arguments and some large ones
        exparr = np.exp(np.minimum(np.maximum(arr, lowx), bigx))
        elowx = exp(lowx)
        darr_low = lowx - arr
        ebigx = exp(bigx)
        darr_big = arr - bigx
        denom = 1.0 + darr_low * (1.0 + 0.5 * darr_low)
        exparr_smaller = elowx * (1.0 + darr_low) / (denom * denom)
        exparr_larger = ebigx * (1.0 + darr_big)
        if verbose:
            _report_small_args(arr, lowx, bs_name_func())
            _report_large_args(arr, bigx, bs_name_func())
        resus = exparr
        resus[arr < lowx] = exparr_smaller[arr < lowx]
        resus[arr > bigx] = exparr_larger[arr > bigx]
        return resus


def der2_npexp(arr: np.ndarray, bigx: Optional[float] = 100.0, lowx: Optional[float] = -100.0,
               verbose: Optional[bool] = False) -> np.ndarray:
    """
    second derivative of :math:`C^2` extension of  :math:`\\exp(a)` above `bigx` and below `lowx`

    :param np.ndarray arr: a Numpy array

    :param float bigx: upper bound

    :param float lowx: lower bound

    :param verbose: if True, reports diagnosis

    :return: second derivative of :math:\\exp(a)`  :math:`C^2`-extended above `bigx`  and below `lowx`
    """
    min_arr, max_arr = np.min(arr), np.max(arr)
    if max_arr <= bigx and min_arr >= lowx:
        return np.exp(arr)
    elif max_arr > bigx and min_arr > lowx:  # some large arguments, no small ones
        exparr = np.exp(np.minimum(arr, bigx))
        ebigx = exp(bigx)
        exparr_larger = ebigx
        if verbose:
            _report_large_args(arr, bigx, bs_name_func())
        return np.where(arr < bigx, exparr, exparr_larger)
    elif max_arr < bigx and min_arr < lowx:  # some small arguments, no large ones
        exparr = np.exp(np.maximum(arr, lowx))
        elowx = exp(lowx)
        darr = lowx - arr
        denom = 1.0 + darr * (1.0 + 0.5 * darr)
        exparr_smaller = elowx * (1.0 + darr) / (denom * denom)
        if verbose:
            _report_small_args(arr, lowx, bs_name_func())
        return np.where(arr > lowx, exparr, exparr_smaller)
    else:  # some small arguments and some large ones
        exparr = np.exp(np.minimum(np.maximum(arr, lowx), bigx))
        elowx = exp(lowx)
        darr_low = lowx - arr
        ebigx = exp(bigx)
        darr_big = arr - bigx
        denom = 1.0 + darr_low * (1.0 + 0.5 * darr_low)
        exparr_smaller = elowx * (1.0 + darr_low) / (denom * denom)
        exparr_larger = ebigx * (1.0 + darr_big)
        if verbose:
            _report_small_args(arr, lowx, bs_name_func())
            _report_large_args(arr, bigx, bs_name_func())
        resus = exparr
        resus[arr < lowx] = exparr_smaller[arr < lowx]
        resus[arr > bigx] = exparr_larger[arr > bigx]
        return resus


def nppow(a: np.ndarray, b: Union[int, float, np.ndarray]) -> np.ndarray:
    """
    evaluates a**b element-by-element

    :param np.ndarray a: an array of any shape

    :param Union[int, float, np.ndarray] b: if an array,
       should have the same shape as `a`

    :return: an array of the same shape as `a`
    """
    mina = np.min(a)
    if mina <= 0:
        print_stars("All elements of a must be positive in nppow!")
        sys.exit(1)

    if isinstance(b, (int, float)):
        return a ** b
    else:
        if a.shape != b.shape:
            print_stars(
                "nppow: b is not a number or an array of the same shape as a!")
            sys.exit(1)
        avec = a.ravel()
        bvec = b.ravel()
        a_pow_b = avec ** bvec
        return a_pow_b.reshape(a.shape)


def der_nppow(a: np.ndarray, b: Union[int, float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    evaluates the derivatives in a and b of element-by-element a**b

    :param np.ndarray a: an array of any shape

    :param Union[int, float, np.ndarray] b: if an array,
       should have the same shape as `a`

    :return: a pair of two arrays of the same shape as `a`
    """

    mina = np.min(a)
    if mina <= 0:
        print_stars("All elements of a must be positive in der_nppow!")
        sys.exit(1)

    if isinstance(b, (int, float)):
        a_pow_b = a ** b
        return (b * a_pow_b / a, a_pow_b * np.log(a))
    else:
        if a.shape != b.shape:
            print_stars(
                "nppow: b is not a number or an array of the same shape as a!")
            sys.exit(1)
        avec = a.ravel()
        bvec = b.ravel()
        a_pow_b = avec ** bvec
        der_wrt_a = a_pow_b * bvec / avec
        der_wrt_b = a_pow_b * nplog(avec)
        return der_wrt_a.reshape(a.shape), der_wrt_b.reshape(a.shape)


def der2_nppow(a: np.ndarray, b: Union[int, float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    evaluates the second derivatives in (a,a), (a,b), and (b,b) of element-by-element a**b

    :param np.ndarray a: an array of any shape

    :param Union[int, float, np.ndarray] b: if an array,
       should have the same shape as `a`

    :return: a 3-uple of  arrays of the same shape as `a`
    """

    mina = np.min(a)
    if mina <= 0:
        print_stars("All elements of a must be positive in der2_nppow!")
        sys.exit(1)

    if isinstance(b, (int, float)):
        b1 = b - 1.0
        a_pow_b = a ** b
        a_pow_b1 = a_pow_b / a
        log_a = np.log(a)
        return b * b1 * a_pow_b1 / a, b * a_pow_b1 * log_a, a_pow_b * log_a * log_a
    else:
        if a.shape != b.shape:
            print_stars(
                "nppow: b is not a number or an array of the same shape as a!")
            sys.exit(1)
        avec = a.ravel()
        bvec = b.ravel()
        a_pow_b = avec ** bvec
        a_pow_b1 = a_pow_b / avec
        b1 = bvec - 1.0
        log_avec = nplog(avec)
        der2_wrt_aa = bvec * b1 * a_pow_b1 / avec
        der2_wrt_ab = a_pow_b1 * (1.0 + bvec * log_avec)
        der2_wrt_bb = a_pow_b * log_avec * log_avec
        return der2_wrt_aa.reshape(a.shape), der2_wrt_ab.reshape(a.shape), der2_wrt_bb.reshape(a.shape)


def nprepeat_col(v: np.ndarray, n: int) -> np.ndarray:
    """
    create a matrix with `n` columns equal to `v`

    :param  np.array v: a 1-dim array of size `m`

    :param int n: number of columns requested

    :return: a 2-dim array of shape `(m, n)`
    """
    return np.repeat(v[:, np.newaxis], n, axis=1)


def nprepeat_row(v: np.ndarray, m: int) -> np.ndarray:
    """
    create a matrix with `m` rows equal to `v`

    :param np.array v: a 1-dim array of size `n`

    :param int m: number of rows requested

    :return: a 2-dim array of shape `(m, n)`
    """
    return np.repeat(v[np.newaxis, :], m, axis=0)


def npmaxabs(arr: np.ndarray) -> float:
    """
    maximum absolute value in an array

    :param np.array arr: Numpy array

    :return: a float
    """
    return np.max(np.abs(arr))


def d2log(X: Union[float, np.ndarray], dX: np.ndarray, d2X: np.ndarray) -> np.ndarray:
    """
    evaluates the Hessian of log(X) from the values and derivatives of X

    :param X: scalar or array of dimension 1 or 2

    :param np.ndarray dX: its gradients

    :param np.ndarray d2X: its Hessian

    :return: the Hessian of log(X)
    """
    if isinstance(X, float):
        dlogX = dX / X
        return d2X / X - np.outer(dlogX, dlogX)
    elif isinstance(X, np.ndarray):
        if X.ndim == 1:
            dlogX = dX / X.reshape((-1, 1))
            nX, n_params = dX.shape
            d2lX = np.empty((nX, n_params, n_params))
            for ipar in range(n_params):
                d2lX[:, ipar, :] = d2X[:, ipar, :] / X.reshape((-1, 1)) \
                                   - dlogX * dlogX[:, ipar].reshape((-1, 1))
            return d2lX
        if X.ndim == 2:
            nX1, nX2, n_params = dX.shape
            d2lX = np.empty((nX1, nX2, n_params, n_params))
            for ipar1 in range(n_params):
                for ipar2 in range(ipar1, n_params):
                    d2lX[:, :, ipar1, ipar2] = d2X[:, :, ipar1, ipar2] / X \
                                               - (dX[:, :, ipar1] * dX[:, :, ipar2]) / (X * X)
                    d2lX[:, :, ipar2, ipar1] = d2lX[:, :, ipar1, ipar2]
            return d2lX
        else:
            bs_error_abort(f"not implemented for X of shape {X.shape}")
    else:
        bs_error_abort("only implemented for X scalar or 1- or 2-dimensional")
