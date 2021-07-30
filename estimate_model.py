"""
estimate a model
"""

import numpy as np
from typing import List, Optional, Union, Tuple, Dict

from knitro.numpy.knitroNumPy import *

from cupid_classes import CupidParams, CupidParamsCSHeteroxy, CupidParamsFcmnl
from cupid_utils import print_stars, GRADIENT_STEP, bs_error_abort
from cupid_optim_utils import print_optimization_results, minimize_some_fixed, print_optimization_results_MPEC

from log_likelihood import log_likelihood, grad_log_likelihood
from log_likelihood_MPEC import log_likelihood_fcmnl_MPEC, grad_log_likelihood_fcmnl_MPEC


def maximize_loglik(model_params: Union[CupidParams, CupidParamsCSHeteroxy, CupidParamsFcmnl],
                    x_init: np.ndarray,
                    lower: Optional[np.ndarray] = None,
                    upper: Optional[np.ndarray] = None,
                    checkgrad: Optional[bool] = False,
                    verbose: Optional[bool] = False,
                    fixed_vars: Optional[List[int]] = None,
                    fixed_vals: Optional[List[float]] = None,
                    options: Optional[Dict] = {'iprint': 1}) -> Tuple[float, np.ndarray, int]:
    """
    estimates a model by maximizing the log-likelihood

    :param model_params: the model we estimate

    :param x_init: initial values of parameters

    :param lower: lower bounds on parameters

    :param upper: upper bounds on parameters

    :param boolean checkgrad: if True we check the anaytical gradient

    :param boolean verbose: if True, we print stuff

    :param List[int] fixed_vars: indices of coefficients we fix

    :param List[float] fixed_vals: values at which we fix them

    :param options: passed on to scipy.optimize.minimize

    :return: a 3-uple with the value of the log-likelihood, the estimates, and the convergence code
    """
    n_params = x_init.size
    try:
        kc = KN_new()
    except:
        bs_error_abort("Failed to find a valid Knitro license.")

    KN_add_vars(kc, n_params)

    # bounds, if any
    if lower is None:
        # not necessary since infinite
        KN_set_var_lobnds(kc, xLoBnds=np.full(n_params, -KN_INFINITY))
    else:
        KN_set_var_lobnds(kc, xLoBnds=lower)
    if upper is None:
        KN_set_var_upbnds(kc, xUpBnds=np.full(n_params, KN_INFINITY))
    else:
        KN_set_var_upbnds(kc, xUpBnds=upper)

    # Define an initial point.  If not set, Knitro will generate one.
    KN_set_var_primal_init_values(kc, xInitVals=x_init)

    if fixed_vars is not None:
        assert fixed_vals is not None
        KN_set_var_fxbnds(kc, fixed_vars, fixed_vals)

    cb = KN_add_eval_callback(kc, evalObj=True, funcCallback=log_likelihood)

    KN_set_cb_user_params(kc, cb, model_params)

    KN_set_cb_grad(kc, cb, objGradIndexVars=KN_DENSE,
                   gradCallback=grad_log_likelihood)

    KN_set_int_param(kc, KN_PARAM_OUTLEV, KN_OUTLEV_ALL)

    if checkgrad:
        # Perform a derivative check.
        KN_set_int_param(kc, KN_PARAM_DERIVCHECK, KN_DERIVCHECK_ALL)

    # Solve the problem.
    nStatus = KN_solve(kc)

    loglik_val, estimates = print_optimization_results(kc)

    print_stars()
    print(f" Value of log-likelihood: {loglik_val: > 8.3f}\n")
    print()

    return loglik_val, np.array(estimates), nStatus


def maximize_loglik_fcmnl_MPEC(model_params: Union[CupidParams, CupidParamsCSHeteroxy, CupidParamsFcmnl],
                         x_init: np.ndarray,
                         lower: Optional[np.ndarray] = None,
                         upper: Optional[np.ndarray] = None,
                         checkgrad: Optional[bool] = False,
                         verbose: Optional[bool] = False,
                         fixed_vars: Optional[List[int]] = None,
                         fixed_vals: Optional[List[float]] = None,
                         options: Optional[Dict] = {'iprint': 1}) -> Tuple[float, np.ndarray, int]:
    """
    estimates a model by maximizing the log-likelihood under the constraint that U minimizes G+H

    :param model_params: the model we estimate

    :param x_init: initial values of parameters

    :param lower: lower bounds on parameters

    :param upper: upper bounds on parameters

    :param boolean checkgrad: if True we check the anaytical gradient

    :param boolean verbose: if True, we print stuff

    :param List[int] fixed_vars: indices of coefficients we fix

    :param List[float] fixed_vals: values at which we fix them

    :param options: passed on to scipy.optimize.minimize

    :return: a 3-uple with the value of the log-likelihood, the estimates, and the convergence code
    """
    n_paramsU = x_init.size
    bases_surplus = model_params.bases_surplus
    ncat_men, ncat_women, n_bases = bases_surplus.shape
    n_pars_b_men, n_pars_b_women = model_params.n_pars_b_men, model_params.n_pars_b_women
    n_pars_b = n_pars_b_men + n_pars_b_women
    n_thetas = n_pars_b + n_bases

    try:
        kc = KN_new()
    except:
        bs_error_abort("Failed to find a valid Knitro license.")

    KN_add_vars(kc, n_paramsU)

    # bounds, if any
    if lower is None:
        # not necessary since infinite
        KN_set_var_lobnds(kc, xLoBnds=np.full(n_paramsU, -KN_INFINITY))
    else:
        KN_set_var_lobnds(kc, xLoBnds=lower)
    if upper is None:
        KN_set_var_upbnds(kc, xUpBnds=np.full(n_paramsU, KN_INFINITY))
    else:
        KN_set_var_upbnds(kc, xUpBnds=upper)

    n_prod_categories = ncat_men * ncat_women
    # Add the constraints and set the rhs and coefficients
    n_cons = n_prod_categories
    KN_add_cons(kc, n_cons)
    KN_set_con_eqbnds(kc, cEqBnds=[0.0] * n_cons)

    # Define an initial point.  If not set, Knitro will generate one.
    KN_set_var_primal_init_values(kc, xInitVals=x_init)

    if fixed_vars is not None:
        assert fixed_vals is not None
        KN_set_var_fxbnds(kc, fixed_vars, fixed_vals)

    cb = KN_add_eval_callback(kc, evalObj=True, indexCons=range(n_prod_categories),
                              funcCallback=log_likelihood_fcmnl_MPEC)

    KN_set_cb_user_params(kc, cb, model_params)

    n_args_jac = n_thetas + ncat_men + ncat_women
    n_jac = n_cons * n_args_jac
    cbjacIndexCons = np.arange(n_prod_categories, n_args_jac)
    cbjacIndexVars = np.zeros(n_prod_categories * n_args_jac, int)
    i = 0
    for iman in range(ncat_men):
        for iwoman in range(ncat_women):
            # derivatives in thetas
            cbjacIndexVars[i:(i+n_thetas)] = np.arange(n_thetas)
            # derivatives in [iman, jwoman]
            cbjacIndexVars[(i + n_thetas):(i+n_thetas+ncat_women)] = \
                np.arange(iman*ncat_women,  (iman+1)*ncat_women)
            cbjacIndexVars[(i + n_thetas + ncat_women):(i+n_args_jac)] = \
                iwoman + ncat_women*np.arange(ncat_men)
            i += n_args_jac

    KN_set_cb_grad(objGradIndexVars=KN_DENSE,
                   jacIndexCons=cbjacIndexCons, jacIndexVars=cbjacIndexVars,
                   gradCallback=grad_log_likelihood_fcmnl_MPEC)

    KN_set_int_param(kc, KN_PARAM_OUTLEV, KN_OUTLEV_ALL)

    if checkgrad:
        # Perform a derivative check.
        KN_set_int_param(kc, KN_PARAM_DERIVCHECK, KN_DERIVCHECK_ALL)

    # Solve the problem.
    nStatus = KN_solve(kc)

    loglik_val, estimates = print_optimization_results_MPEC(kc)

    print_stars()
    print(f" Value of log-likelihood: {loglik_val: > 8.3f}\n")
    print()

    return loglik_val, np.array(estimates), nStatus
