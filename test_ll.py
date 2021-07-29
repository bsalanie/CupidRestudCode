"""
tests the log-likelihood and its gradient
"""
import numpy as np

from cupid_classes import MatchingMus, CupidParamsFcmnl
from cupid_utils import GRADIENT_STEP, print_stars, describe_array
from cupid_math_utils import bslog, der_bslog
from cupid_numpy_utils import nplog, der_nplog
from fcmnl import make_b8
from solve_for_mus import mus_fcmnl_and_maybe_grad_agd
from ipfp_solvers import ipfp_homo_solver

def test_log_likelihood(params, model_params):
    """

    :param CupidParamsXXX model_params: the model data

    :return: the value of the loglikelihood
    """
    mus_and_maybe_grad = model_params.mus_and_maybe_grad
    observed_matching = model_params.observed_matching
    mus, U = mus_and_maybe_grad(params, model_params, gr=False)
    obj = -test_loglik_mus(observed_matching, mus)

    #    print(f"log_mus = {eval_result.obj}")

    return obj


def test_grad_log_likelihood(params, model_params):
    """

    :param CupidParamsXXX model_params: the model data

    :return: the gradient of the loglikelihood
    """

    mus_and_maybe_grad = model_params.mus_and_maybe_grad
    bases_surplus = model_params.bases_surplus
    observed_matching = model_params.observed_matching

    ncat_men, ncat_women = bases_surplus.shape[:-1]
    n_prod_categories = ncat_men * ncat_women

    mus, _, dmus = mus_and_maybe_grad(params, model_params, gr=True)

    grad_loglik = test_grad_loglik_all_mus(observed_matching, mus)

    gradN = grad_loglik[-1]
    gradxy = grad_loglik[:n_prod_categories].reshape(
        (ncat_men, ncat_women)) + gradN
    gradx0 = grad_loglik[n_prod_categories:(
            n_prod_categories + ncat_men)] + gradN
    grad0y = grad_loglik[(n_prod_categories + ncat_men):-1] + gradN

    der_muxy = np.einsum('ij,ijk->k', gradxy, dmus.muxy)
    der_mux0 = np.einsum('i,ik->k', gradx0, dmus.mux0)
    der_mu0y = np.einsum('i,ik->k', grad0y, dmus.mu0y)

    objGrad = -(der_muxy + der_mux0 + der_mu0y)

    n_pars_b = 4
    print("grad_log_mus : done")
    for i in range(n_pars_b):
        print(f"Coeff[{i}] = {params[i]: > 10.3f}, gradient: {objGrad[i]: > 10.3f}")

    return objGrad


def test_loglik_mus(observed_matching, simulated_matching):
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


def test_grad_loglik_all_mus(observed_matching, simulated_matching):
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


if __name__ == "__main__":

    # we generate a Choo and Siow homo matching
    ncat_men = ncat_women = 25
    n_prod_categories = ncat_men * ncat_women

    n_bases = 4
    bases_surplus = np.zeros((ncat_men, ncat_women, n_bases))
    x_men = (np.arange(ncat_men) - ncat_men / 2.0) / ncat_men
    y_women = (np.arange(ncat_women) - ncat_women / 2.0) / ncat_women

    bases_surplus[:, :, 0] = 1
    for iy in range(ncat_women):
        bases_surplus[:, iy, 1] = x_men
    for ix in range(ncat_men):
        bases_surplus[ix, :, 2] = y_women
    for ix in range(ncat_men):
        for iy in range(ncat_women):
            bases_surplus[ix, iy, 3] = (x_men[ix] - y_women[iy]) \
                * (x_men[ix] - y_women[iy])

    np.random.seed(7875674)
    men_margins = np.random.uniform(1.0, 10.0, size=ncat_men)
    women_margins = np.random.uniform(1.0, 10.0, size=ncat_women)

    true_surplus_params = np.array([3.0, -1.0, -1.0, -2.0])
    true_surplus_matrix = bases_surplus @ true_surplus_params

    observed_matching, _, _ = ipfp_homo_solver(true_surplus_matrix, men_margins, women_margins)

    EPS = GRADIENT_STEP

    print_stars("Checking the FCMNL model")

    make_b = make_b8
    par_b_men = [0.1, 0.1]
    par_b_women = [0.1, 0.1]

    pars_b_men_arr = np.array(par_b_men)
    pars_b_women_arr = np.array(par_b_women)
    n_pars_b_men = pars_b_men_arr.size
    n_pars_b_women = pars_b_women_arr.size
    n_pars_b = n_pars_b_men + n_pars_b_women

    model_params = CupidParamsFcmnl(men_margins=men_margins, women_margins=women_margins,
                                    bases_surplus=bases_surplus,
                                    n_pars_b_men=n_pars_b_men,
                                    n_pars_b_women=n_pars_b_women,
                                    make_b=make_b,
                                    observed_matching=observed_matching,
                                    mus_and_maybe_grad=mus_fcmnl_and_maybe_grad_agd, tol_agd=1e-9)

    n_params = n_pars_b + n_bases
    params = np.zeros(n_params)
    params[:n_pars_b_men] = pars_b_men_arr
    params[n_pars_b_men:n_pars_b] = pars_b_women_arr

    ll0 = test_log_likelihood(params, model_params)
    gradll = test_grad_log_likelihood(params, model_params)

    gradll_num = np.zeros_like(gradll)

    for ipar in range(n_params):
        par1 = params.copy()
        par1[ipar] += EPS
        ll1 = test_log_likelihood(par1, model_params)
        gradll_num[ipar] = (ll1 - ll0) / EPS
        print(f"{ipar}: num = {gradll_num[ipar]}, anal={gradll[ipar]}")



