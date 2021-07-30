"""
defines the Knitro callbacks for :math:`\\log(L)`: and the :math:`\\nabla G =\\nabla_V H` constraint, MPEC version
"""

from knitro.numpy.knitroNumPy import *

from cupid_classes import MatchingMus, CupidParamsFcmnl
from fcmnl import derivs_GplusH_fcmnl

from log_likelihood import loglik_mus, grad_loglik_all_mus


def log_likelihood_fcmnl_MPEC(kc, cb, eval_request, eval_result, model_params):
    """
    this is the Knitro callback for the  MPEC loglikelihood

    :param CupidParamsFcmnl model_params: the model data

    :return: nothing
    """
    if eval_request.type != KN_RC_EVALFC:
        print("*** log_likelihood_fcmnl_MPEC incorrectly called with eval type %d" %
              eval_request.type)
        return -1

    paramsU = eval_request.x
    bases_surplus = model_params.bases_surplus
    ncat_men, ncat_women, n_bases = bases_surplus.shape
    n_pars_b_men, n_pars_b_women = model_params.n_pars_b_men, model_params.n_pars_b_women
    n_pars_b = n_pars_b_men + n_pars_b_women
    n_thetas = n_pars_b + n_bases
    thetas = paramsU[:n_thetas]
    pars_b_men = thetas[:n_pars_b_men]
    pars_b_women = thetas[n_pars_b_men: n_pars_b]
    pars_bases = thetas[n_pars_b:]
    U = paramsU[n_thetas:]

    Phi = bases_surplus @ pars_bases

    # mu is the gradient of G wrt U
    resus_ders = derivs_GplusH_fcmnl(U, model_params, Phi, pars_b_men, pars_b_women,
                                     derivs=1)
    muxy = resus_ders.gradients[1]  # this is mu = G_U
    mux0 = model_params.men_margins - np.sum(muxy, 1)
    mu0y = model_params.women_margins - np.sum(muxy, 0)
    mus = MatchingMus(muxy, mux0, mu0y)

    eval_result.obj = -loglik_mus(model_params.observed_matching, mus)

    n_prod_categories = ncat_women * ncat_women

    # grad(G)-grad(H)
    eval_result.c = (muxy - resus_ders.gradients[2]).reshape(n_prod_categories)

    return 0


def grad_log_likelihood_fcmnl_MPEC(kc, cb, eval_request, eval_result, model_params):
    if eval_request.type != KN_RC_EVALGA:
        print("*** grad_log_likelihood_fcmnl_MPEC incorrectly called with eval type %d" %
              eval_request.type)
        return -1
    params = eval_request.x

    np.savetxt("current_pars_m.txt", params)

    paramsU = eval_request.x
    bases_surplus = model_params.bases_surplus
    ncat_men, ncat_women, n_bases = bases_surplus.shape
    n_pars_b_men, n_pars_b_women = model_params.n_pars_b_men, model_params.n_pars_b_women
    n_pars_b = n_pars_b_men + n_pars_b_women
    n_thetas = n_pars_b + n_bases
    thetas = paramsU[:n_thetas]
    pars_b_men = thetas[:n_pars_b_men]
    pars_b_women = thetas[n_pars_b_men: n_pars_b]
    pars_bases = thetas[n_pars_b:]
    U = paramsU[n_thetas:]

    Phi = bases_surplus @ pars_bases
    resus_hess = derivs_GplusH_fcmnl(U, model_params, Phi, pars_b_men, pars_b_women,
                                     derivs=2)
    muxy = resus_hess.gradients[1]
    mux0 = model_params.men_margins - np.sum(muxy, 1)
    mu0y = model_params.women_margins - np.sum(muxy, 0)
    mus = MatchingMus(muxy, mux0, mu0y)
    d2G_dU, d2G_UU, d2H_dV, d2H_VV = resus_hess.hessians

    observed_matching = model_params.observed_matching

    n_prod_categories = ncat_men * ncat_women

    grad_loglik = grad_loglik_all_mus(observed_matching, mus)

    gradN = grad_loglik[-1]
    gradxy = grad_loglik[:n_prod_categories].reshape(
        (ncat_men, ncat_women))
    gradx0 = grad_loglik[n_prod_categories:(
            n_prod_categories + ncat_men)]
    grad0y = grad_loglik[(n_prod_categories + ncat_men):-1]

    # the total derivative in muxy
    gradxy -= grad0y
    gradxy -= gradx0.reshape((-1,1))
    gradxy -= gradN

    n_paramsU = n_thetas + n_prod_categories
    grad = np.zeros(n_paramsU)
    for ipar in range(n_pars_b_men):
        # derivative of muxy in ipar
        dmuxy_dth = d2G_dU[ipar, :].reshape((ncat_men, ncat_women))
        grad[ipar] = np.sum(gradxy * dmuxy_dth)
    for ipar in range(n_prod_categories):
        dmuxy_dth = d2G_UU[ipar, :].reshape((ncat_men, ncat_women))
        grad[ipar + n_thetas] = np.sum(gradxy * dmuxy_dth)

    eval_result.objGrad = -grad

    # Evaluate nonlinear terms in constraint gradients (Jacobian)
    jac_thetas = d2G_dU - d2H_dV
    jac_U = d2G_UU + d2H_VV
    for ipar in range(n_thetas):
        phib = bases_surplus[:, :, ipar].reshape(n_prod_categories)
        jac_thetas[ipar, :] -= (d2H_VV @ phib)

    eval_result.jac = 0

    return 0
