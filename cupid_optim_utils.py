""" helpful routines for minimization """

import numpy as np
from typing import Callable, Optional, Tuple, Any, List, Dict, Union
from math import sqrt
import scipy.linalg as spla

from knitro.numpy.knitroNumPy import *


from cupid_utils import print_stars, bs_error_abort
from cupid_numpy_utils import npmaxabs


def print_optimization_results(kc) -> None:
    """
    print results from unconstrained optimization

    :param kc: Knitro controller

    :return: nothing
    """
    # get solution information.
    nStatus, objSol, x, lambdas = KN_get_solution(kc)
    grads = KN_get_objgrad_values_all(kc)

    print_stars(f"Knitro ended with nStatus={nStatus}")
    print("  feasibility violation    = %e" % KN_get_abs_feas_error(kc))
    print("  KKT optimality violation = %e" % KN_get_abs_opt_error(kc))

    loglik_val = -objSol
    print_stars()
    print(f" value of loglikelihood: {loglik_val: > 8.3f}\n")
    print()

    print_stars("Coefficients,    gradients,    multipliers")
    i = 1
    for estimate, gradval, lambdaval in zip(x, grads, lambdas):
        print(
            f"[{i}]:  {estimate: > 10.3f},       {gradval: > 10.3f},    {lambdaval: > 10.3f}")
        i += 1

    return loglik_val, x


def _identity_prox_h(x: np.ndarray, t: float, p: np.ndarray) -> np.ndarray:
    """ trivial proximal projector, for use in :fun:`acc_grad_descent`"""
    return x


def acc_grad_descent(grad_f: Callable, x_init: np.ndarray,
                     prox_h: Optional[Callable] = None, other_params: Optional[Any] = None,
                     print_result: Optional[bool] = False, verbose: Optional[bool] = False,
                     tol: Optional[float] = 1e-9, alpha: Optional[float] = 1.01, beta: Optional[float] = 0.5,
                     maxiter: Optional[int] = 10000) -> Tuple[np.ndarray, int]:
    """
    minimizes :math:`(f+h)` by Accelerated Gradient Descent
     where `f` is smooth and convex  and `h` is convex.

    By default `h` is zero.

    :param Callable grad_f: grad_f of `f`; should return an `(n)` array from an `(n)` array \
       and the `other_ params` object

    :param np.array x_init: initial guess, shape `(n)`

    :param Callable prox_h: proximal projector of `h`, if any; should return an `(n)` array from \
    an `(n)` array, a float, and an `(n)` array

    :param other_params: an object with additional parameters passed on to the gradient of :math:`f`

    :param bool print_result: if `True`, print the minimization result

    :param bool verbose: if `True`, print diagnosis

    :param float tol: convergence criterion on absolute grad_f

    :param float alpha: ceiling on step multiplier

    :param float beta: floor on step multiplier

    :param int maxiter: max number of iterations

    :return: candidate solution, and 1 if converged/0 if not
    """

    # no proximal projection if no h
    local_prox_h = prox_h if prox_h else _identity_prox_h

    x = x_init.copy()
    y = x_init.copy()

    g = grad_f(y, other_params)
    theta = 1.0
    #  for stepsize we use Barzilai-Borwein
    t = 1.0 / spla.norm(g)
    x_hat = x - t * g
    g_hat = grad_f(x_hat, other_params)
    norm_dg = spla.norm(g - g_hat)
    norm_dg2 = norm_dg * norm_dg
    t = np.abs(np.dot(x - x_hat, g - g_hat)) / norm_dg2

    grad_err_init = npmaxabs(g)

    if verbose:
        print(f"agd: grad_err_init={grad_err_init}")

    iterations = 0

    while iterations < maxiter:
        grad_err = npmaxabs(g)
        if grad_err < tol:
            break
        xi = x
        yi = y
        x = y - t * g
        x = local_prox_h(x, t, other_params)

        theta = 2.0 / (1.0 + sqrt(1.0 + 4.0 / theta / theta))

        if np.dot(y - x, x - xi) > 0:  # wrong direction, we restart
            x = xi
            y = x
            theta = 1.0
        else:
            y = x + (1.0 - theta) * (x - xi)

        gi = g
        g = grad_f(y, other_params)
        ndy = spla.norm(y - yi)
        t_hat = 0.5 * ndy * ndy / abs(np.dot(y - yi, gi - g))
        t = min(alpha * t, max(beta * t, t_hat))

        iterations += 1

        if verbose:
            print(
                f" AGD with grad_err = {grad_err} after {iterations} iterations")

    x_conv = y

    ret_code = 0 if grad_err < tol else 1

    if verbose or print_result:
        if ret_code == 0:
            print_stars(
                f" AGD converged with grad_err = {grad_err} after {iterations} iterations")
        else:
            print_stars(
                f" Problem in AGD: grad_err = {grad_err} after {iterations} iterations")

    return x_conv, ret_code


def _fix_some(obj: Callable, grad_obj: Callable,
              fixed_vars: List[int], fixed_vals: np.ndarray) -> Tuple[Callable, Callable]:
    """
    Takes in a function and its gradient, fixes the variables
    whose indices are `fixed_vars` to the values in `fixed_vals`, 
    and returns the modified function and its gradient

    :param obj: the original function

    :param grad_obj: its gradient function

    :param fixed_vars: a list if the indices of variables whose values are fixed

    :param fixed_vals: their fixed values

    :return: the modified function and its modified gradient function

    """

    def fixed_obj(t, other_args):
        t_full = list(t)
        for i, i_coef in enumerate(fixed_vars):
            t_full.insert(i_coef, fixed_vals[i])
        arr_full = np.array(t_full)
        return obj(arr_full, other_args)

    def fixed_grad_obj(t, other_args):
        t_full = list(t)
        for i, i_coef in enumerate(fixed_vars):
            t_full.insert(i_coef, fixed_vals[i])
        arr_full = np.array(t_full)
        grad_full = grad_obj(arr_full, other_args)
        return np.delete(grad_full, fixed_vars)

    return fixed_obj, fixed_grad_obj


def minimize_some_fixed(obj: Callable, grad_obj: Callable,
                        x_init: np.ndarray, args: List,
                        fixed_vars: Union[List[int], None], fixed_vals: Union[np.ndarray, None],
                        options: Optional[Dict] = None,
                        bounds: Optional[List[Tuple]] = None):
    """
    minimize a function with some variables fixed, using L-BFGS-B

    :param obj: the original function

    :param grad_obj: its gradient function

    :param fixed_vars: a list if the indices of variables whose values are fixed

    :param fixed_vals: their fixed values

    :param x_init: the initial values of all variables (those on fixed variables are not used)

    :param args: a list of other parameters

    :param options: any options passed on to scipy.optimize.minimize

    :param bounds: the bounds on all variables (those on fixed variables are not used)

    :return: the result of optimization, on all variables
    """
    if fixed_vars is None:
        resopt = spopt.minimize(obj, x_init, method='L-BFGS-B',
                                args=args, options=options,
                                jac=grad_obj, bounds=bounds)
    else:
        if len(fixed_vars) != fixed_vals.size:
            bs_error_abort(
                f"fixed_vars has {len(fixed_vars)} indices but fixed_vals has {fixed_vals.size} elements.")
        fixed_obj, fixed_grad_obj = _fix_some(
            obj, grad_obj, fixed_vars, fixed_vals)

        # drop fixed variables and the corresponding bounds
        n = len(x_init)
        not_fixed = np.ones(n, dtype=bool)
        not_fixed[fixed_vars] = False
        t_init = x_init[not_fixed]
        t_bounds = [bounds[i] for i in range(n) if not_fixed[i]]

        resopt = spopt.minimize(fixed_obj, t_init, method='L-BFGS-B',
                                args=args, options=options,
                                jac=fixed_grad_obj, bounds=t_bounds)

        # now re-fill the values of the variables
        t = resopt.x
        t_full = list(t)
        for i, i_coef in enumerate(fixed_vars):
            t_full.insert(i_coef, fixed_vals[i])
        resopt.x = t_full

        # and re-fill the values of the gradients
        g = grad_obj(np.array(t_full), args)
        resopt.jac = g

    return resopt


if __name__ == "__main__":
    def obj(x, args):
        res = x - args
        return np.sum(res * res)

    def grad_obj(x, args):
        res = x - args
        return 2.0 * res

    n = 5
    x_init = np.zeros(n)
    args = np.arange(n)
    bounds = [(-10.0, 10.0) for _ in range(n)]

    fixed_vars = [1, 3]
    fixed_vals = -np.ones(2)

    resopt = minimize_some_fixed(obj, grad_obj, x_init, args,
                                 fixed_vars=fixed_vars, fixed_vals=fixed_vals,
                                 bounds=bounds)

    print_optimization_results(resopt)
