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


def analyze_results_fcmnl():
    plots_dir = root_dir + "Plots/"

    # where we stored the results
    fcmnl_dir = root_dir + "Results/Fcmnl/"

    fcmnl_dir_homo = fcmnl_dir + "ReplicatingChooSiowHomo/"
    str_fcmnl_homo = "homovecx"
    str_title_homo = "Choo-Siow homo"

    fcmnl_dir1 = fcmnl_dir + "Exponent1/"
    str_fcmnl1 = "newvat1"
    str_title1 = "FC-MNL b/|x-z|"

    fcmnl_dir2 = fcmnl_dir + "Exponent2/"
    str_fcmnl2 = "newvat2"
    str_title2 = "FC-MNL b/|x-z|^2"

    fcmnl_dir1x = fcmnl_dir + "Exponent1x/"
    str_fcmnl1x = "newvecx"
    str_title1x = "FC-MNL b(y)/|x-z|"

    str_fcmnl = [str_fcmnl_homo, str_fcmnl1, str_fcmnl2, str_fcmnl1x]
    str_titles = [str_title_homo, str_title1, str_title2, str_title1x]
    dirs_fcmnl = [fcmnl_dir_homo, fcmnl_dir1, fcmnl_dir2, fcmnl_dir1x]
    npars_fcmnl = [30, 31, 32, 32]
    npars_b = [0, 2, 4, 4]

    resus = {}
    for i, name in enumerate(str_fcmnl):
        fits = np.loadtxt(dirs_fcmnl[i] + "Fcmnl_" + name + "_fits.txt")
        # total loglik
        fits[0] *= N_HOUSEHOLDS_OBS
        # recompute AIC, BIC
        fits[1] = fits[0] - npars_fcmnl[i]
        fits[2] = fits[0] - npars_fcmnl[i] * log(N_HOUSEHOLDS_OBS) / 2.0
        resi = fits
        resus[name] = resi

    dres = pd.DataFrame(index=str_titles, columns=['Total logL', 'AIC', 'BIC'])

    dres.loc[str_title_homo, :] = resus[str_fcmnl_homo]
    dres.loc[str_title1, :] = resus[str_fcmnl1]
    dres.loc[str_title2, :] = resus[str_fcmnl2]
    dres.loc[str_title1x, :] = resus[str_fcmnl1x]

    pd.set_option('precision', 9)
    pd.set_option('display.max_columns', 5)
    print_stars("Fits for FCMNL")
    print(dres)

    for ispecif in range(1, len(str_titles)):
        print_stars("Specification " + str_titles[ispecif])
        estimates = np.loadtxt(dirs_fcmnl[ispecif] + "Fcmnl_" +
                               str_fcmnl[ispecif] + "_thetas.txt")
        np_b = npars_b[ispecif]
        np_b2 = int(np_b / 2)
        print_stars("Estimated b")
        print(f"Men: {estimates[:np_b2]}")
        print(f"Women: {estimates[np_b2:np_b]}")

    hessG_homo = np.loadtxt(fcmnl_dir_homo + "Fcmnl_" + str_fcmnl_homo + "_d2G_UU.txt")
    hessH_homo = np.loadtxt(fcmnl_dir_homo + "Fcmnl_" + str_fcmnl_homo + "_d2H_VV.txt")
    # for the homogeneous case we need to renormalize the utility scale
    tau = 1.1
    hessG_homo /= tau
    hessH_homo /= tau

    hessG1 = np.loadtxt(fcmnl_dir1 + "Fcmnl_" + str_fcmnl1 + "_d2G_UU.txt")
    hessH1 = np.loadtxt(fcmnl_dir1 + "Fcmnl_" + str_fcmnl1 + "_d2H_VV.txt")
    hessG2 = np.loadtxt(fcmnl_dir2 + "Fcmnl_" + str_fcmnl2 + "_d2G_UU.txt")
    hessH2 = np.loadtxt(fcmnl_dir2 + "Fcmnl_" + str_fcmnl2 + "_d2H_VV.txt")
    hessG1x = np.loadtxt(fcmnl_dir1x + "Fcmnl_" + str_fcmnl1x + "_d2G_UU.txt")
    hessH1x = np.loadtxt(fcmnl_dir1x + "Fcmnl_" + str_fcmnl1x + "_d2H_VV.txt")

    # need to regroup the nonzero elements
    ncat_men = ncat_women = 25
    n_prod_categories = ncat_men * ncat_women
    d2G_homo = []
    d2G1 = []
    d2G2 = []
    d2G1x = []
    ivar = 0
    for iman in range(ncat_men):
        slice_man = slice(ivar, ivar + ncat_women)
        d2G_homo.append(hessG_homo[slice_man, slice_man])
        d2G1.append(hessG1[slice_man, slice_man])
        d2G2.append(hessG2[slice_man, slice_man])
        d2G1x.append(hessG1x[slice_man, slice_man])
        ivar += ncat_women

    d2H_homo = []
    d2H1 = []
    d2H2 = []
    d2H1x = []
    # V was [iman, iwoman]
    for iwoman in range(ncat_women):
        slice_woman = slice(iwoman, n_prod_categories, ncat_women)
        d2H_homo.append(hessH_homo[slice_woman, slice_woman])
        d2H1.append(hessH1[slice_woman, slice_woman])
        d2H2.append(hessH2[slice_woman, slice_woman])
        d2H1x.append(hessH1x[slice_woman, slice_woman])

    # now each d2G[x] is a (ncat_women, ncat_women) matrix  of dmu(y|x)/dU(xt)
    #  and each d2H[y] is a (ncat_men, ncat_men) matrix  of dmu(x|y)/dV(zy)

    # read the simulated mus
    muxy_norm_homo = np.loadtxt(fcmnl_dir_homo + "Fcmnl_" + str_fcmnl_homo
                                + "_muxy_norm.txt")
    mux0_norm_homo = np.loadtxt(fcmnl_dir_homo + "Fcmnl_"
                                + str_fcmnl_homo + "_mux0_norm.txt")
    mu0y_norm_homo = np.loadtxt(fcmnl_dir_homo + "Fcmnl_"
                                + str_fcmnl_homo + "_mu0y_norm.txt")
    muxy_norm1 = np.loadtxt(fcmnl_dir1 + "Fcmnl_" + str_fcmnl1 + "_muxy_norm.txt")
    mux0_norm1 = np.loadtxt(fcmnl_dir1 + "Fcmnl_" + str_fcmnl1 + "_mux0_norm.txt")
    mu0y_norm1 = np.loadtxt(fcmnl_dir1 + "Fcmnl_" + str_fcmnl1 + "_mu0y_norm.txt")
    muxy_norm2 = np.loadtxt(fcmnl_dir2 + "Fcmnl_" + str_fcmnl2 + "_muxy_norm.txt")
    mux0_norm2 = np.loadtxt(fcmnl_dir2 + "Fcmnl_" + str_fcmnl2 + "_mux0_norm.txt")
    mu0y_norm2 = np.loadtxt(fcmnl_dir2 + "Fcmnl_" + str_fcmnl2 + "_mu0y_norm.txt")
    muxy_norm1x = np.loadtxt(fcmnl_dir1x + "Fcmnl_" + str_fcmnl1x + "_muxy_norm.txt")
    mux0_norm1x = np.loadtxt(fcmnl_dir1x + "Fcmnl_" + str_fcmnl1x + "_mux0_norm.txt")
    mu0y_norm1x = np.loadtxt(fcmnl_dir1x + "Fcmnl_" + str_fcmnl1x + "_mu0y_norm.txt")

    nx_homo = np.sum(muxy_norm_homo, 1) + mux0_norm_homo
    my_homo = np.sum(muxy_norm_homo, 0) + mu0y_norm_homo
    mugivenx_homo = muxy_norm_homo / nx_homo.reshape((-1, 1))
    mugiveny_homo = muxy_norm_homo / my_homo
    nx1 = np.sum(muxy_norm1, 1) + mux0_norm1
    my1 = np.sum(muxy_norm1, 0) + mu0y_norm1
    mugivenx1 = muxy_norm1 / nx1.reshape((-1, 1))
    mugiveny1 = muxy_norm1 / my1
    nx2 = np.sum(muxy_norm2, 1) + mux0_norm2
    my2 = np.sum(muxy_norm2, 0) + mu0y_norm2
    mugivenx1 = muxy_norm2 / nx2.reshape((-1, 1))
    mugiveny2 = muxy_norm2 / my2
    nx1x = np.sum(muxy_norm1x, 1) + mux0_norm1x
    my1x = np.sum(muxy_norm1x, 0) + mu0y_norm1x
    mugivenx1x = muxy_norm1x / nx1x.reshape((-1, 1))
    mugiveny1x = muxy_norm1x / my1x

    # check for the homogeneous case
    d2G_homo_th = []
    for iman in range(ncat_men):
        muxy_man = muxy_norm_homo[iman, :]
        mugivenx_man = mugivenx_homo[iman, :]
        d2G_th = np.diag(muxy_man) - np.outer(muxy_man, mugivenx_man)
        d2G_homo_th.append(d2G_th)

    d2H_homo_th = []
    for iwoman in range(ncat_women):
        muxy_woman = muxy_norm_homo[:, iwoman]
        mugiveny_woman = mugiveny_homo[:, iwoman]
        d2H_th = np.diag(muxy_woman) - np.outer(muxy_woman, mugiveny_woman)
        d2H_homo_th.append(d2H_th)

    print_stars("Checking d2G for the homogeneous case: ratio should be 1")
    for iman in range(ncat_men):
        d2G_ratio_x = d2G_homo[iman] / d2G_homo_th[iman]
        print(f"x = {iman}: min = {np.min(d2G_ratio_x): > 6.3f}, "
              f"max = {np.max(d2G_ratio_x): > 6.3f}")

    print_stars("Checking d2H for the homogeneous case: ratio should be 1")
    for iwoman in range(ncat_women):
        d2H_ratio_y = d2H_homo[iwoman] / d2H_homo_th[iwoman]
        print(f"y = {iwoman}: min = {np.min(d2H_ratio_y): > 6.3f}, "
              f"max = {np.max(d2H_ratio_y): > 6.3f}")

    print_stars("Now ratio of " + str_title1 + " to CS homo")
    d2H_fratio1 = []
    for iwoman in range(ncat_women):
        d2H_fratio1.append(d2H1[iwoman] / d2H_homo[iwoman])
    print_stars(" and ratio of " + str_title2 + " to CS homo")
    d2H_fratio2 = []
    for iwoman in range(ncat_women):
        d2H_fratio2.append(d2H2[iwoman] / d2H_homo[iwoman])

    # plotting
    import matplotlib.pyplot as plt

    # from mpl_toolkits.axes_grid1 import make_axes_locatable

    # style
    plt.style.use('seaborn')

    plt.clf()

    nrows = 2
    ncols = 3

    def plot_matching(axi, fratio, iwoman, ncat_women):
        np.fill_diagonal(fratio, 0.0)
        age_min = 16 + max(0, iwoman - 5)
        age_max = 16 + min(ncat_women, iwoman + 5)
        n_ages = age_max - age_min
        slice_woman = slice(age_min - 16, age_max - 16 + 1)
        axi.imshow(fratio[slice_woman, slice_woman],
                   origin='lower', cmap="YlOrRd")
        ages_coord = [0, n_ages]
        ages = [str(age_min), str(age_max)]
        axi.set_xticks(ages_coord)
        axi.set_xticklabels(ages)
        axi.set_yticks(ages_coord)
        axi.set_yticklabels(ages)
        axi.set_title("Age " + str(iwoman + 16))

    fig, ax = plt.subplots(nrows, ncols, constrained_layout=True)
    iman = 0
    for i in range(nrows):
        for j in range(ncols):
            fratio = d2G_homo[iman] / muxy_norm_homo[iman, :].reshape((-1, 1))
            plot_matching(ax[i, j], fratio, iman, ncat_men)
            iman += 1

    # fig.suptitle("Semi-elasticities for men, " + str_title_homo)
    plt.savefig(plots_dir + "Semi_elasticities_Fcmnl_men_" + str_fcmnl_homo + ".eps")

    plt.clf()

    fig, ax = plt.subplots(nrows, ncols, constrained_layout=True)
    iwoman = 0
    for i in range(nrows):
        for j in range(ncols):
            fratio = d2H_homo[iwoman] / muxy_norm_homo[:, iwoman].reshape((-1, 1))
            plot_matching(ax[i, j], fratio, iwoman, ncat_women)
            iwoman += 1

    # fig.suptitle("Semi-elasticities for women, " + str_title_homo)
    plt.savefig(plots_dir + "Semi_elasticities_Fcmnl_women_" + str_fcmnl_homo + ".eps")

    plt.clf()

    fig, ax = plt.subplots(nrows, ncols, constrained_layout=True)
    iwoman = 0
    for i in range(nrows):
        for j in range(ncols):
            fratio = d2H1[iwoman] / muxy_norm1[:, iwoman].reshape((-1, 1))
            plot_matching(ax[i, j], fratio, iwoman, ncat_women)
            iwoman += 1

    # fig.suptitle("Semi-elasticities for women, " + str_title1)
    plt.savefig(plots_dir + "Semi_elasticities_Fcmnl_women_" + str_fcmnl1 + ".eps")

    plt.clf()

    fig, ax = plt.subplots(nrows, ncols, constrained_layout=True)
    iwoman = 0
    for i in range(nrows):
        for j in range(ncols):
            fratio = d2H2[iwoman] / muxy_norm2[:, iwoman].reshape((-1, 1))
            plot_matching(ax[i, j], fratio, iwoman, ncat_women)
            iwoman += 1

    # fig.suptitle("Semi-elasticities for women, " + str_title2)
    plt.savefig(plots_dir + "Semi_elasticities_Fcmnl_women_" + str_fcmnl2 + ".eps")

    plt.clf()

    fig, ax = plt.subplots(nrows, ncols, constrained_layout=True)
    iman = 0
    for i in range(nrows):
        for j in range(ncols):
            fratio = d2G1x[iman] / muxy_norm1x[iman, :].reshape((-1, 1))
            plot_matching(ax[i, j], fratio, iman, ncat_men)
            iman += 1

    # fig.suptitle("Semi-elasticities for men, " + str_title1x)
    plt.savefig(plots_dir + "Semi_elasticities_Fcmnl_men_" + str_fcmnl1x + ".eps")

    plt.clf()

    fig, ax = plt.subplots(nrows, ncols, constrained_layout=True)
    iwoman = 0
    for i in range(nrows):
        for j in range(ncols):
            fratio = d2H1x[iwoman] / muxy_norm1x[:, iwoman].reshape((-1, 1))
            plot_matching(ax[i, j], fratio, iwoman, ncat_women)
            iwoman += 1

    # fig.suptitle("Semi-elasticities for women, " + str_title1x)
    plt.savefig(plots_dir + "Semi_elasticities_Fcmnl_women_" + str_fcmnl1x + ".eps")
