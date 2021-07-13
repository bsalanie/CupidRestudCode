"""
analyze  estimates
"""
import numpy as np
import scipy.linalg as spla
from math import log
import pandas as pd
import multiprocessing as mp
from typing import Optional, List
from pathlib import Path

from cupid_classes import MatchingMus, ModelParams
from cupid_utils import root_dir, N_HOUSEHOLDS_OBS, print_stars, eval_moments
from cupid_numpy_utils import d2log
from log_likelihood import loglik_mus


def compute_Imat(params: np.ndarray, mus: MatchingMus, dmus: MatchingMus, sumw2: MatchingMus,
                 do_center: Optional[bool] = True) -> np.ndarray:
    """
    compute the outer product of the scores I, with sampling weights,  dividing by N_HOUSEHOLDS_OBS

    :param np.ndarray params: estimated parameter values

    :param MatchingMus mus: estimated (muxy, mux0, mu0y)

    :param MatchingMus dmus: their derivatives wrt the params

    :param MatchingMus sumw2: sums of squared sampling weights in each cell

    :param boolean do_center: True if we center derivatives

    :return: the I matrix
    """
    n_params = params.size

    muxy = mus.muxy
    mux0 = mus.mux0
    mu0y = mus.mu0y
    dmus_xy = dmus.muxy
    dmus_x0 = dmus.mux0
    dmus_0y = dmus.mu0y
    N = np.sum(muxy) + np.sum(mux0) + np.sum(mu0y)
    dN = np.einsum('ijk->k', dmus_xy) + np.sum(dmus_x0, 0) + np.sum(dmus_0y, 0)
    dlogN = dN / N

    # compute the dlog(mu/N)/dtheta
    muxy = mus.muxy
    dlogmuxy_N = np.empty_like(dmus_xy)
    dlogmux0_N = np.empty_like(dmus_x0)
    dlogmu0y_N = np.empty_like(dmus_0y)
    for ik in range(n_params):
        dlogmuxy_N[:, :, ik] = dmus_xy[:, :, ik] / muxy - dlogN[ik]
        dlogmux0_N[:, ik] = dmus_x0[:, ik] / mux0 - dlogN[ik]
        dlogmu0y_N[:, ik] = dmus_0y[:, ik] / mu0y - dlogN[ik]

    # weights
    sumw2_xy, sumw2_x0, sumw2_0y = sumw2.unpack()

    if do_center:
        # first, center
        mean_dlogmuxy_N = np.einsum('ijk,ij->k', dlogmuxy_N, sumw2_xy) / np.sum(sumw2_xy)
        mean_dlogmux0_N = np.einsum('ik,i->k', dlogmux0_N, sumw2_x0) / np.sum(sumw2_x0)
        mean_dlogmu0y_N = np.einsum('ik,i->k', dlogmu0y_N, sumw2_0y) / np.sum(sumw2_0y)
        for ik in range(n_params):
            dlogmuxy_N[:, :, ik] -= mean_dlogmuxy_N[ik]
            dlogmux0_N[:, ik] -= mean_dlogmux0_N[ik]
            dlogmu0y_N[:, ik] -= mean_dlogmu0y_N[ik]

    term_xy = np.einsum('ij,ijk,ijl->kl', sumw2_xy, dlogmuxy_N, dlogmuxy_N)
    term_x0 = np.einsum('i,ik,il->kl', sumw2_x0, dlogmux0_N, dlogmux0_N)
    term_0y = np.einsum('j,jk,jl->kl', sumw2_0y, dlogmu0y_N, dlogmu0y_N)

    Imat = (term_xy + term_x0 + term_0y) / (np.sum(sumw2_xy)
                                            + np.sum(sumw2_x0)
                                            + np.sum(sumw2_0y))

    return Imat


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


def analyze_results(model_params: ModelParams, estimates: np.ndarray, sumw2: MatchingMus, str_model: str,
                    results_dir: Path, exclude_coeffs: Optional[List[int]] = None,
                    do_stderrs: Optional[bool] = False, save: Optional[bool] = False):
    """

    :param CupidParams model_params: the model we estimated

    :param np.ndarray estimates: the estimated parameters

    :param MatchingMus sumw2: sums of squared sampling weights in each cell

    :param str str_model: a title

    :param str results_dir: directory where we save the results

    :param exclude_coeffs: if True, we exclude these coefficients in the computation of the standard errors

    :param boolean do_stderrs: if True, we compute the values of the standard errors

    :param boolean save: if True, we save the results

    :return: nothing
    """

    bases_surplus = model_params.bases_surplus

    mu_hat_norm = model_params.observed_matching
    mus_and_maybe_grad = model_params.mus_and_maybe_grad

    simulated_matching_norm, U_conv, dmus = \
        mus_and_maybe_grad(estimates, model_params, gr=True)
    simulated_moments_norm = eval_moments(simulated_matching_norm.muxy,
                                          bases_surplus)

    if save:
        np.savetxt(results_dir + str_model + "_thetas.txt", estimates)
        muxy = simulated_matching_norm.muxy
        mux0 = simulated_matching_norm.mux0
        mu0y = simulated_matching_norm.mu0y
        np.savetxt(results_dir + str_model + "_muxy_norm.txt", muxy)
        np.savetxt(results_dir + str_model + "_mux0_norm.txt", mux0)
        np.savetxt(results_dir + str_model + "_mu0y_norm.txt", mu0y)
        np.savetxt(results_dir + str_model + "_U.txt", U_conv)

    moments_hat_norm = eval_moments(mu_hat_norm.muxy, bases_surplus)
    print(f"Observed normalized moments: {moments_hat_norm}")
    print(f"    simulated normalized moments: {simulated_moments_norm}")

    print(f"Observed normalized muxy: {mu_hat_norm.muxy[:4, :4]}")
    print(f"Simulated normalized muxy: {simulated_matching_norm.muxy[:4, :4]}")

    print(f"Observed normalized mux0: {mu_hat_norm.mux0}")
    print(f"Simulated normalized mux0: {simulated_matching_norm.mux0}")

    print(f"Observed normalized mu0y: {mu_hat_norm.mu0y}")
    print(f"Simulated normalized mu0y: {simulated_matching_norm.mu0y}")

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
        np.savetxt(results_dir + str_model + "_fits.txt", fits)

    if do_stderrs:
        Imat = compute_Imat(estimates, simulated_matching_norm,
                            dmus, sumw2)

        Jmat = compute_Jmat(estimates, model_params,
                            simulated_matching_norm, dmus)

        if save:
            np.savetxt(results_dir + str_model + "_Jmat.txt", Jmat)
            np.savetxt(results_dir + str_model + "_Imat.txt", Imat)

        if exclude_coeffs is not None:
            # kludge for badly identified coefficients
            for i in exclude_coeffs:
                Imat[i, :] = 0.0
                Imat[:, i] = 0.0
                Imat[i, i] = 1.0
                Jmat[i, :] = 0.0
                Jmat[:, i] = 0.0
                Jmat[i, i] = 1.0

        invJ = spla.inv(Jmat)
        J1_I_J1 = invJ @ (Imat @ invJ)
        varcov = J1_I_J1 / N_HOUSEHOLDS_OBS
        if exclude_coeffs is not None:
            for i in exclude_coeffs:
                varcov[i, :] = 0.0
                varcov[:, i] = 0.0
                varcov[i, i] = 1.0
                stderrs = np.sqrt(np.diag(varcov))

        estimates_stderrs = np.column_stack((estimates, stderrs))
        print_stars("estimated coefficients and standard errors")
        for i in range(n_params):
            print(f"{i + 1: > 3d}: {estimates[i]: > 10.3f}     ({stderrs[i]: > 10.3f})")
        np.savetxt(results_dir + str_model + "_estimates.txt", estimates_stderrs)

