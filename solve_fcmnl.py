import numpy as np

from knitro.numpy.knitroNumPy import *

from cupid_classes import MatchingMus, CupidParamsFcmnl
from cupid_utils import GRADIENT_STEP, print_stars, describe_array, bs_error_abort
from cupid_math_utils import bslog, der_bslog
from cupid_numpy_utils import nplog, der_nplog
from fcmnl import make_b8, derivs_GplusH_fcmnl

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

    print("\n ENTERING F")

    U = eval_request.x

    model_params, Phi, pars_b_men, pars_b_women = other_params

    GandH = derivs_GplusH_fcmnl(U, model_params, Phi, pars_b_men, pars_b_women, derivs=0)
    Gval, Hval = GandH.values
    eval_result.obj = Gval + Hval

    print("\n EXITING F")

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

    print("\n ENTERING G")

    model_params, Phi, pars_b_men, pars_b_women = other_params

    GandHwithgradients = derivs_GplusH_fcmnl(U, model_params, Phi, pars_b_men, pars_b_women, derivs=1)

    gradG_d, gradG_U, gradH_d, gradH_V = GandHwithgradients.gradients

    eval_result.objGrad = gradG_U - gradH_V

    print("\n EXITING G")

    return 0



def hess_GplusH_fcmnl(kc, cb, eval_request, eval_result, other_params):
    """
    this is the Knitro callback for the hessian of G(U) + H(Phi-U)

    :param other_params: the model data, Phi, and the FCMNL parameters

    :return: nothing
    """
    if eval_request.type != KN_RC_H and evalRequest.type != KN_RC_EVALH_NO_F:
        print("*** hess_GplusH_fcmnl incorrectly called with eval type %d" %
              eval_request.type)
        return -1
    U = eval_request.x

    print("\n ENTERING H")

    model_params, Phi, pars_b_men, pars_b_women = other_params

    GandHwithhessians = derivs_GplusH_fcmnl(U, model_params, Phi, pars_b_men, pars_b_women, derivs=2)

    d2G_dU, d2G_UU, d2H_dV, d2H_VV = GandHwithhessians.hessians
    
    hessianwrtU = d2G_UU+d2H_VV
     
   # we only give the upper triangle, row-major
    n_prod_categories = hessianwrtU.shape[0]
    n_hess = (n_prod_categories*(n_prod_categories+1))/2
    hessian = np.zeros(n_hess)
    
    k = 0
    for i in range(n_prod_categories):
        for j in range(i, n_prod_categories):
            hessian[k] = hessianwrtU[i, j]

    eval_result.objHess = hessian

    print("\n EXITING H")

    return 0


def minimize_fcmnl_U(U_init, other_params, checkgrad=False, verbose=False):
    print("\n    entered inner  \n")
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

    KN_set_cb_hess(kc, cb, hessIndexVars1 = KN_DENSE_ROWMAJOR,
                   hessCallback = hess_GplusH_fcmnl)

    KN_set_int_param(kc, KN_PARAM_CONVEX, KN_CONVEX_YES)

    KN_set_int_param(kc, KN_PARAM_ALGORITHM, KN_ALG_BAR_DIRECT)

    KN_set_int_param(kc, KN_PARAM_OUTLEV, KN_OUTLEV_ALL)

    if checkgrad:
        # Perform a derivative check.
        KN_set_int_param(kc, KN_PARAM_DERIVCHECK, KN_DERIVCHECK_ALL)

    # Solve the problem.
    nStatus = KN_solve(kc)

    print(f"\n\ndone inner,  status {nStatus}\n")

    # get solution information.
    nStatus, objSol, U_conv, lambdas = KN_get_solution(kc)

    return U_conv, nStatus

