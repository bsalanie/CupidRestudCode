import numpy as np

from knitro.numpy.knitroNumPy import *

from cupid_classes import MatchingMus, CupidParamsFcmnl
from cupid_utils import GRADIENT_STEP, print_stars, describe_array, bs_error_abort
from cupid_math_utils import bslog, der_bslog
from cupid_numpy_utils import nplog, der_nplog
from fcmnl import make_b8, derivs_GplusH_fcmnl
from solve_for_mus import mus_fcmnl_and_maybe_grad_agd
from ipfp_solvers import ipfp_homo_solver
from collections import namedtuple


def GplusH_fcmnl(kc, cb, eval_request, eval_result, other_params):
    """
    this is the Knitro callback for  G(U) + H(Phi-U)

    :param other_params: the model data, Phi, and the FCMNL parameters

    :return: nothing
    """
    if eval_request.type != KN_RC_EVALFC:
        print("*** GplusH_fcmnl incorrectly called with eval type %d" %
              eval_request.type)
        return -1
    U = eval_request.x

    model_params, Phi, pars_b_men, pars_b_women = other_params

    GandH = derivs_GplusH_fcmnl(U, model_params, Phi, pars_b_men, pars_b_women, derivs=0)
    Gval, Hval = GandH.values
    eval_result.obj = Gval + Hval

    return 0


def grad_GplusH_fcmnl(kc, cb, eval_request, eval_result, other_params):
    """
    this is the Knitro callback for the gradient of G(U) + H(Phi-U)

    :param other_params: the model data, Phi, and the FCMNL parameters

    :return: nothing
    """
    if eval_request.type != KN_RC_EVALGA:
        print("*** grad_GplusH_fcmnl incorrectly called with eval type %d" %
              eval_request.type)
        return -1
    U = eval_request.x

    model_params, Phi, pars_b_men, pars_b_women = other_params

    GandHwithgradients = derivs_GplusH_fcmnl(U, model_params, Phi, pars_b_men, pars_b_women, derivs=1)

    gradG_d, gradG_U, gradH_d, gradH_V = GandHwithgradients.gradients

    eval_result.objGrad = gradG_U - gradH_V

    return 0


def minimize_fcmnl_U(U_init, other_params, checkgrad=False, verbose=False):
    n_prod_categories = U_init.size
    try:
        kc = KN_new()
    except:
        bs_error_abort("Failed to find a valid Knitro license.")

    KN_add_vars(kc, n_prod_categories)

    # Define an initial point.  If not set, Knitro will generate one.
    KN_set_var_primal_init_values(kc, xInitVals=U_init)

    cb = KN_add_eval_callback(kc, evalObj=True, funcCallback=GplusH_fcmnl)

    KN_set_cb_user_params(kc, cb, other_params)

    KN_set_cb_grad(kc, cb, objGradIndexVars=KN_DENSE,
                   gradCallback=grad_GplusH_fcmnl)

    KN_set_int_param(kc, KN_PARAM_OUTLEV, KN_OUTLEV_ALL)

    if checkgrad:
        # Perform a derivative check.
        KN_set_int_param(kc, KN_PARAM_DERIVCHECK, KN_DERIVCHECK_ALL)

    # Solve the problem.
    nStatus = KN_solve(kc)

    # get solution information.
    nStatus, objSol, U_conv, lambdas = KN_get_solution(kc)

    return U_conv, nStatus

