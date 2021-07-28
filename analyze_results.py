"""
analyze  estimates
"""
import numpy as np
import scipy.linalg as spla
from math import log

import multiprocessing as mp
from typing import Optional, List
from pathlib import Path

from cupid_classes import MatchingMus, ModelParams
from cupid_utils import N_HOUSEHOLDS_OBS, print_stars, eval_moments, mkdir_if_needed
from cupid_numpy_utils import d2log
from log_likelihood import loglik_mus




def compute_Jmat(params: np.ndarray, model_params: ModelParams, mus: MatchingMus, dmus: MatchingMus) -> np.ndarray:
    """
    compute the second derivative J, with sampling weights, dividing by N_HOUSEHOLDS_OBS

    :param np.ndarray params: estimated parameter values

    :param CupidParams model_params: model parameters

    :param MatchingMus mus: estimated (muxy, mux0, mu0y)

    :param dmus: their derivatives wrt the params

    :return: the J matrix
    """
    mus_and_maybe_grad = model_params.mus_and_maybe_grad
    bases_surplus = model_params.bases_surplus
    observed_matching = model_params.observed_matching
    muxy_hat = observed_matching.muxy
    mux0_hat = observed_matching.mux0
    mu0y_hat = observed_matching.mu0y

    ncat_men, ncat_women, n_bases = bases_surplus.shape

    n_params = params.size
    muxy = mus.muxy
    mux0 = mus.mux0
    mu0y = mus.mu0y
    dmus_xy = dmus.muxy
    dmus_x0 = dmus.mux0
    dmus_0y = dmus.mu0y
    N = np.sum(muxy) + np.sum(mux0) + np.sum(mu0y)
    dN = np.einsum('ijk->k', dmus_xy) + np.sum(dmus_x0, 0) + np.sum(dmus_0y, 0)

    d2mus_xy = np.empty((ncat_men, ncat_women, n_params, n_params))
    d2mus_x0 = np.empty((ncat_men, n_params, n_params))
    d2mus_0y = np.empty((ncat_women, n_params, n_params))

    GRADIENT_STEP = 1e-6

    def d2row(ipar):
        """ evaluate row ipar of the second derivatives matrices """
        params1 = params.copy()
        params1[ipar] = params[ipar] + GRADIENT_STEP
        _, _, dmus_p = mus_and_maybe_grad(params1, model_params, gr=True)
        params1[ipar] = params[ipar] - GRADIENT_STEP
        _, _, dmus_m = mus_and_maybe_grad(params1, model_params, gr=True)
        d2mus_xy = (dmus_p.muxy - dmus_m.muxy) / (2.0 * GRADIENT_STEP)
        d2mus_x0 = (dmus_p.mux0 - dmus_m.mux0) / (2.0 * GRADIENT_STEP)
        d2mus_0y = (dmus_p.mu0y - dmus_m.mu0y) / (2.0 * GRADIENT_STEP)
        return MatchingMus(d2mus_xy, d2mus_x0, d2mus_0y)

    use_mp = False

    n_cpus = mp.cpu_count()
    if use_mp and n_cpus > 1:
        use_cpus = n_cpus - 1

        args_mp = []
        for ipar in range(n_params):
            args_mp.append([ipar])
        with mp.Pool(processes=use_cpus) as pool:
            res = pool.starmap(d2row, args_mp)
    else:
        res = []
        for ipar in range(n_params):
            res.append(d2row(ipar))

    for ipar in range(n_params):
        resi = res[ipar]
        d2mus_xy[:, :, ipar, :] = resi.muxy
        d2mus_x0[:, ipar, :] = resi.mux0
        d2mus_0y[:, ipar, :] = resi.mu0y

    d2N = np.einsum('ijkl->kl', d2mus_xy) + np.einsum('ikl->kl', d2mus_x0) \
          + np.einsum('jkl->kl', d2mus_0y)

    d2logN = d2log(N, dN, d2N)
    d2logmuxy = d2log(muxy, dmus_xy, d2mus_xy)
    d2logmux0 = d2log(mux0, dmus_x0, d2mus_x0)
    d2logmu0y = d2log(mu0y, dmus_0y, d2mus_0y)

    d2logmuxy_N = d2logmuxy - d2logN
    d2logmux0_N = d2logmux0 - d2logN
    d2logmu0y_N = d2logmu0y - d2logN

    Jmat = -(np.einsum('ijkl,ij->kl', d2logmuxy_N, muxy_hat)
             + np.einsum('ikl,i->kl', d2logmux0_N, mux0_hat)
             + np.einsum('jkl,j->kl', d2logmu0y_N, mu0y_hat))

    Jmat /= (np.sum(muxy_hat) + np.sum(mux0_hat) + np.sum(mu0y_hat))

    return Jmat


def analyze_results(model_params: ModelParams, estimates: np.ndarray, 
                    str_model: str,
                    results_dir: Path, do_stderrs: Optional[bool] = False,
                    varmus: Optional[np.ndarray] = None, save: Optional[bool] = False):
    """

    :param CupidParams model_params: the model we estimated

    :param np.ndarray estimates: the estimated parameters

    :param str str_model: a title

    :param str results_dir: directory where we save the results

    :param boolean do_stderrs: if True, we compute the values of the standard errors

    :param np.ndarray varmus: then we need the variance of the observed mus

    :param boolean save: if True, we save the results

    :return: nothing
    """

    bases_surplus = model_params.bases_surplus

    ncat_men, ncat_women, n_bases = bases_surplus.shape
    n_prod_categories = ncat_men*ncat_women
        
    mu_hat_norm = model_params.observed_matching
    mus_and_maybe_grad = model_params.mus_and_maybe_grad

    simulated_matching_norm, U_conv, dmus = \
        mus_and_maybe_grad(estimates, model_params, gr=True)
    simulated_moments_norm = eval_moments(simulated_matching_norm.muxy,
                                          bases_surplus)

    muxy = simulated_matching_norm.muxy
    mux0 = simulated_matching_norm.mux0
    mu0y = simulated_matching_norm.mu0y

    if save:
        results_model = mkdir_if_needed(results_dir / str_model)
        np.savetxt(results_model / "thetas.txt", estimates)
        np.savetxt(results_model / "muxy_norm.txt", muxy)
        np.savetxt(results_model / "mux0_norm.txt", mux0)
        np.savetxt(results_model / "mu0y_norm.txt", mu0y)
        np.savetxt(results_model / "U.txt", U_conv)

    moments_hat_norm = eval_moments(mu_hat_norm.muxy, bases_surplus)
    print(f"Observed normalized moments: {moments_hat_norm}")
    print(f"    simulated normalized moments: {simulated_moments_norm}")

    print(f"Observed normalized muxy: {mu_hat_norm.muxy[:4, :4]}")
    print(f"Simulated normalized muxy: {muxy[:4, :4]}")

    print(f"Observed normalized mux0: {mu_hat_norm.mux0}")
    print(f"Simulated normalized mux0: {mux0}")

    print(f"Observed normalized mu0y: {mu_hat_norm.mu0y}")
    print(f"Simulated normalized mu0y: {mu0y}")

    loglik_val = loglik_mus(mu_hat_norm, simulated_matching_norm)
    print(f"The loglikelihood is {loglik_val:.6f} per obs")

    loglik_val_tot = N_HOUSEHOLDS_OBS * loglik_val
    print(f"The loglikelihood is {loglik_val_tot:.6f} in toto")

    n_params = estimates.size
    AIC_val = -2.0 * loglik_val_tot + 2.0 * n_params
    BIC_val = -2.0 * loglik_val_tot + log(N_HOUSEHOLDS_OBS) * n_params
    print(f"The AIC is {AIC_val:.6f}")
    print(f"The BIC is {BIC_val:.6f}")

    if save:
        fits = np.array([loglik_val, AIC_val, BIC_val])
        np.savetxt(results_model / "fits.txt", fits)

    if do_stderrs:
        muxyv = muxy.reshape(n_prod_categories)
        Jmat = compute_Jmat(estimates, model_params,
                            simulated_matching_norm, dmus)
        dlogmuxy_dtheta = np.zeros((n_prod_categories, n_params))
        for i in range(n_params):
            dmuxyi = dmus.muxy[:, :, i].reshape(n_prod_categories)
            dlogmuxy_dtheta[:, i] = dmuxyi/muxyv
        dlogmux0_dtheta = dmus.mux0/mux0.reshape((-1,1))
        dlogmu0y_dtheta = dmus.mu0y/mu0y.reshape((-1,1))
        dlogmu_dtheta = np.concatenate((dlogmuxy_dtheta, dlogmux0_dtheta, dlogmu0y_dtheta))

        dtheta_dmuhat = - spla.solve(Jmat, dlogmu_dtheta.T)

        n_bigvar = n_prod_categories + ncat_men + ncat_women
        bigvarmus = np.zeros((n_bigvar, n_bigvar))
        bigvarmus[:n_prod_categories, :n_prod_categories] = varmus[0]
        bigvarmus[:n_prod_categories, n_prod_categories:(n_prod_categories+ncat_men)] = varmus[1]
        bigvarmus[n_prod_categories:(n_prod_categories+ncat_men), :n_prod_categories] = varmus[1].T
        bigvarmus[:n_prod_categories, (n_prod_categories+ncat_men):] = varmus[2]
        bigvarmus[(n_prod_categories+ncat_men):, :n_prod_categories] = varmus[2].T
        bigvarmus[n_prod_categories:(n_prod_categories+ncat_men), n_prod_categories:(n_prod_categories+ncat_men)] \
            = varmus[3]
        bigvarmus[n_prod_categories:(n_prod_categories + ncat_men):, (n_prod_categories+ncat_men):] \
            = varmus[4]
        bigvarmus[(n_prod_categories + ncat_men):, n_prod_categories:(n_prod_categories + ncat_men):] \
            = varmus[4].T
        bigvarmus[(n_prod_categories + ncat_men):, (n_prod_categories+ncat_men):] = varmus[5]

        var_theta = dtheta_dmuhat @ (bigvarmus @ dtheta_dmuhat.T)
        stderrs = np.sqrt(np.diag(var_theta))
        students = estimates/stderrs


    if save:
        estimates_stderrs = np.column_stack((estimates, stderrs, students))
        np.savetxt(results_model / "estimates.txt", estimates_stderrs)
        print_stars("estimated coefficients   (standard errors)  [Students]")
        for i in range(n_params):
            print(f"{i+1: 3d}: {estimates[i]: > 10.3f}     ({stderrs[i]: > 10.3f})  [{students[i]: > 10.3f}]")
