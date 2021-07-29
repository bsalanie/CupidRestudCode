"""
defines the Knitro callbacks for :math:`\\log(L)`:  the log-likelihood and its gradient 
"""
import numpy as np

from knitro.numpy.knitroNumPy import *

from cupid_classes import MatchingMus
from cupid_math_utils import bslog, der_bslog
from cupid_numpy_utils import nplog, der_nplog


def log_likelihood(kc, cb, eval_request, eval_result, model_params):
    """
    this is the Knitro callback for the loglikelihood

    :param CupidParamsXXX model_params: the model data

    :return: the value of the loglikelihood
    """
    if eval_request.type != KN_RC_EVALFC:
        print("*** log_mus incorrectly called with eval type %d" %
              eval_request.type)
        return -1
    params = eval_request.x
    mus_and_maybe_grad = model_params.mus_and_maybe_grad
    observed_matching = model_params.observed_matching
    mus, U = mus_and_maybe_grad(params, model_params, gr=False)
    eval_result.obj = -loglik_mus(observed_matching, mus)

#    print(f"log_mus = {eval_result.obj}")

    return 0


def grad_log_likelihood(kc, cb, eval_request, eval_result, model_params):
    """
     this is the Knitro callback for the gradient of the loglikelihood

     :param CupidParamsXXX model_params: the model data

     :return: the gradient of the loglikelihood
     """
    if eval_request.type != KN_RC_EVALGA:
        print("*** grad_log_mus incorrectly called with eval type %d" %
              eval_request.type)
        return -1
    params = eval_request.x

    mus_and_maybe_grad = model_params.mus_and_maybe_grad
    bases_surplus = model_params.bases_surplus
    observed_matching = model_params.observed_matching

    ncat_men, ncat_women = bases_surplus.shape[:-1]
    n_prod_categories = ncat_men * ncat_women

    mus, _, dmus = mus_and_maybe_grad(params, model_params, gr=True)

    grad_loglik = grad_loglik_all_mus(observed_matching, mus)

    gradN = grad_loglik[-1]
    gradxy = grad_loglik[:n_prod_categories].reshape(
        (ncat_men, ncat_women)) + gradN
    gradx0 = grad_loglik[n_prod_categories:(
        n_prod_categories + ncat_men)] + gradN
    grad0y = grad_loglik[(n_prod_categories + ncat_men):-1] + gradN

    der_muxy = np.einsum('ij,ijk->k', gradxy, dmus.muxy)
    der_mux0 = np.einsum('i,ik->k', gradx0, dmus.mux0)
    der_mu0y = np.einsum('i,ik->k', grad0y, dmus.mu0y)

    eval_result.objGrad = -(der_muxy + der_mux0 + der_mu0y)

    n_pars_b = 4
    print("grad_log_mus : done")
    for i in range(n_pars_b):
        print(f"Coeff[{i}] = {params[i]: > 10.3f}, gradient: {eval_result.objGrad[i]: > 10.3f}")

    return 0


def loglik_mus(observed_matching, simulated_matching):
    """
    evaluate the loglikelihood for given matching patterns

    :param MatchingMus observed_matching: observed (muxy, mux0, mu0y)

    :param MatchingMus simulated_matching: simulated (muxy, mux0, mu0y)

    :return: the value of the log-likelihood over the sample
    """

    muxy_sim, mux0_sim, mu0y_sim = simulated_matching.unpack()
    n_households_sim = np.sum(muxy_sim) + np.sum(mux0_sim) + np.sum(mu0y_sim)

    muxy_obs, mux0_obs, mu0y_obs = observed_matching.unpack()
    n_households_obs = np.sum(muxy_obs) + np.sum(mux0_obs) + np.sum(mu0y_obs)

    loglik_value = np.sum(muxy_obs * nplog(muxy_sim)) + \
        np.sum(mux0_obs * nplog(mux0_sim)) + \
        np.sum(mu0y_obs * nplog(mu0y_sim)) - \
        n_households_obs * bslog(n_households_sim)
    return loglik_value


def grad_loglik_all_mus(observed_matching, simulated_matching):
    """
    evaluate the gradient of the loglikelihood wrt (muxy, mux0, mu0y, N) under household sampling, where N is the \
    number of households

    :param MatchingMus observed_matching: observed (muxy, mux0, mu0y)

    :param MatchingMus simulated_matching: simulated (muxy, mux0, mu0y)

    :return: an np.array with the gradient of the log-likelihood
    """

    muxy_sim, mux0_sim, mu0y_sim = simulated_matching.unpack()
    n_households_sim = np.sum(muxy_sim) + np.sum(mux0_sim) + np.sum(mu0y_sim)

    muxy_obs, mux0_obs, mu0y_obs = observed_matching.unpack()
    n_households_obs = np.sum(muxy_obs) + np.sum(mux0_obs) + np.sum(mu0y_obs)

    der_x0 = mux0_obs * der_nplog(mux0_sim)
    der_0y = mu0y_obs * der_nplog(mu0y_sim)
    der_xy = muxy_obs * der_nplog(muxy_sim)
    n_prod_categories, ncat_men, ncat_women = \
        muxy_obs.size, mux0_obs.size, mu0y_obs.size
    grad_loglik = np.zeros(n_prod_categories + ncat_men + ncat_women + 1)
    grad_loglik[:n_prod_categories] = der_xy.reshape(n_prod_categories)
    grad_loglik[n_prod_categories:(n_prod_categories + ncat_men)] \
        = der_x0
    grad_loglik[(n_prod_categories + ncat_men):-1] \
        = der_0y
    grad_loglik[-1] = \
        -n_households_obs * der_bslog(n_households_sim)

    return grad_loglik
