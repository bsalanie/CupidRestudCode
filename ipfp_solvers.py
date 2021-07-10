"""
Implementations of the IPFP algorithm to solve for equilibrium and do comparative statics
in several variants of the `Choo and Siow 2006 <https://www.jstor.org/stable/10.1086/498585?seq=1>`_ model:

all of these variants have discrete variables only

 * homoskedastic with singles (as in CS 2006)
 * gender-heteroskedastic: with a scale parameter on the error term for women
 * gender- and type-heteroskedastic: with a scale parameter on the error term for women

each solver, when fed the joint surplus and margins,
returns the equilibrium matching patterns, the adding-up errors on the margins,
and if requested (gr=True) the derivatives of the matching patterns in all primitives.
"""

import numpy as np
from math import sqrt, log
from typing import Optional, Union, Tuple

import scipy.linalg as spla

from cupid_classes import MatchingMus
from cupid_utils import print_stars, bs_error_abort, describe_array
from cupid_numpy_utils import npexp, der_npexp, npmaxabs, nprepeat_col, nprepeat_row

# the type of the tuple returned by the ipfp_XX routines
IPFPReturn = Union[Tuple[MatchingMus, np.ndarray, np.ndarray],
                   Tuple[MatchingMus, np.ndarray, np.ndarray, MatchingMus]]


def _test_size_Phi(Phi: np.ndarray, ncat_men: int, ncat_women: int):
    if Phi.shape != (ncat_men, ncat_women):
        bs_error_abort(
            f"the shape of Phi should be ({ncat_men}, {ncat_women}), not {Phi.shape}")


def ipfp_homo_solver(Phi: np.ndarray, men_margins: np.ndarray, women_margins: np.ndarray, tol: Optional[float] = 1e-9,
                     gr: Optional[bool] = False, verbose: Optional[bool] = False,
                     maxiter: Optional[int] = 1000) -> IPFPReturn:
    """
    solve for equilibrium in a Choo and Siow market

    given systematic surplus and margins

    :param np.array Phi: matrix of systematic surplus, shape (ncat_men, ncat_women)

    :param np.array men_margins: vector of men margins, shape (ncat_men)

    :param np.array women_margins: vector of women margins, shape (ncat_women)

    :param float tol: tolerance on change in solution

    :param boolean gr: if True, also evaluate derivatives of muxy wrt Phi

    :param boolean verbose: prints stuff

    :param int maxiter: maximum number of iterations

    :return: (muxy, mux0, mu0y), errors on margins marg_err_x, marg_err_y,
             and gradients of (muxy, mux0, mu0y) wrt Phi if gr=True
    """

    ncat_men = men_margins.size
    ncat_women = women_margins.size
    _test_size_Phi(Phi, ncat_men, ncat_women)

    ephi2 = npexp(Phi / 2.0)

    #############################################################################
    # we solve the equilibrium equations muxy = ephi2 * tx * ty
    #   where mux0=tx**2  and mu0y=ty**2
    #   starting with a reasonable initial point for tx and ty: tx = ty = bigc
    #   it is important that the initial point fit the total number of individuals
    #############################################################################

    ephi2T = ephi2.T
    nindivs = np.sum(men_margins) + np.sum(women_margins)
    # calibrate bigc to the total number of individuals
    bigc = sqrt(nindivs / (ncat_men + ncat_women + 2.0 * np.sum(ephi2)))

    txi = np.full(ncat_men, bigc)
    tyi = np.full(ncat_women, bigc)

    err_diff = bigc
    tol_diff = tol * bigc
    niter = 0
    while (err_diff > tol_diff) and (niter < maxiter):
        sx = ephi2 @ tyi
        tx = (np.sqrt(sx * sx + 4.0 * men_margins) - sx) / 2.0
        sy = ephi2T @ tx
        ty = (np.sqrt(sy * sy + 4.0 * women_margins) - sy) / 2.0
        err_x = npmaxabs(tx - txi)
        err_y = npmaxabs(ty - tyi)
        err_diff = err_x + err_y
        txi = tx
        tyi = ty
        niter += 1
    mux0 = txi * txi
    mu0y = tyi * tyi
    muxy = ephi2 * np.outer(txi, tyi)
    marg_err_x = mux0 + np.sum(muxy, 1) - men_margins
    marg_err_y = mu0y + np.sum(muxy, 0) - women_margins
    if verbose:
        print(f"After {niter} iterations:")
        print(f"\tMargin error on x: {npmaxabs(marg_err_x)}")
        print(f"\tMargin error on y: {npmaxabs(marg_err_y)}")
    if not gr:
        return MatchingMus(muxy, mux0, mu0y), marg_err_x, marg_err_y
    else:  # we compute the derivatives
        sxi = ephi2 @ tyi
        syi = ephi2T @ txi
        n_sum_categories = ncat_men + ncat_women
        n_prod_categories = ncat_men * ncat_women
        # start with the LHS of the linear system
        lhs = np.zeros((n_sum_categories, n_sum_categories))
        lhs[:ncat_men, :ncat_men] = np.diag(2.0 * txi + sxi)
        lhs[:ncat_men, ncat_men:] = ephi2 * txi.reshape((-1, 1))
        lhs[ncat_men:, ncat_men:] = np.diag(2.0 * tyi + syi)
        lhs[ncat_men:, :ncat_men] = ephi2T * tyi.reshape((-1, 1))
        # now fill the RHS
        rhs = np.zeros((n_sum_categories, n_prod_categories))
        #  to compute derivatives of (txi, tyi) wrt Phi
        #    this is 1/2 with safeguards
        der_ephi2 = der_npexp(Phi / 2.0) / (2.0 * ephi2)
        ivar = 0
        for iman in range(ncat_men):
            rhs[iman, ivar:(ivar + ncat_women)] = - \
                muxy[iman, :] * der_ephi2[iman, :]
            ivar += ncat_women
        ivar1 = ncat_men
        ivar2 = 0
        for iwoman in range(ncat_women):
            rhs[ivar1, ivar2:n_prod_categories:ncat_women] = - \
                muxy[:, iwoman] * der_ephi2[:, iwoman]
            ivar1 += 1
            ivar2 += 1
        # solve for the derivatives of txi and tyi
        dt_dT = spla.solve(lhs, rhs)
        dt = dt_dT[:ncat_men, :]
        dT = dt_dT[ncat_men:, :]
        # now construct the derivatives of the mus
        dmux0 = 2.0 * (dt * txi.reshape((-1, 1)))
        dmu0y = 2.0 * (dT * tyi.reshape((-1, 1)))
        dmuxy = np.zeros((n_prod_categories, n_prod_categories))
        ivar = 0
        for iman in range(ncat_men):
            dt_man = dt[iman, :]
            dmuxy[ivar:(ivar + ncat_women),
                  :] = np.outer((ephi2[iman, :] * tyi), dt_man)
            ivar += ncat_women
        for iwoman in range(ncat_women):
            dT_woman = dT[iwoman, :]
            dmuxy[iwoman:n_prod_categories:ncat_women,
                  :] += np.outer((ephi2[:, iwoman] * txi), dT_woman)
        # add the term that comes from differentiating ephi2
        muxy_vec2 = (muxy * der_ephi2).reshape(n_prod_categories)
        dmuxy += np.diag(muxy_vec2)
        return MatchingMus(muxy, mux0, mu0y), marg_err_x, marg_err_y, MatchingMus(dmuxy, dmux0, dmu0y)


def ipfp_hetero_solver(Phi: np.ndarray, men_margins: np.ndarray, women_margins: np.ndarray,
                       tau: float, tol: Optional[float] = 1e-9,
                       gr: Optional[bool] = False, verbose: Optional[bool] = False,
                       maxiter: Optional[int] = 1000) -> IPFPReturn:
    """
    solve for equilibrium  in a gender-heteroskedastic Choo and Siow market

    given systematic surplus and margins and a scale parameter dist_params[0]

    :param np.array Phi: matrix of systematic surplus, shape (ncat_men, ncat_women)

    :param np.array men_margins: vector of men margins, shape (ncat_men)

    :param np.array women_margins: vector of women margins, shape (ncat_women)

    :param float tau: the value of the dispersion parameter for women

    :param float tol: tolerance on change in solution

    :param boolean gr: if True, also evaluate derivatives of muxy wrt Phi

    :param boolean verbose: prints stuff

    :param int maxiter: maximum number of iterations

    :return: MatchingMus(muxy, mux0, mu0y), errors on margins marg_err_x, marg_err_y;
             and gradients of (muxy, mux0, mu0y)
             wrt (men_margins, women_margins, Phi, dist_params[0]) if gr=True
    """

    if tau <= 0:
        bs_error_abort(f"needs a positive tau, not {tau}")

    ncat_men = men_margins.shape[0]
    ncat_women = women_margins.shape[0]
    _test_size_Phi(Phi, ncat_men, ncat_women)

    #############################################################################
    # we use ipfp_heteroxy_solver with sigma_x = 1 and tau_y = tau
    #############################################################################

    sigma_x = np.ones(ncat_men)
    tau_y = np.full(ncat_women, tau)

    if not gr:
        return ipfp_heteroxy_solver(Phi, men_margins, women_margins,
                                    sigma_x, tau_y, tol=tol, gr=False,
                                    maxiter=maxiter, verbose=verbose)
    else:
        mus, marg_err_x, marg_err_y, dmus_hxy = \
            ipfp_heteroxy_solver(Phi, men_margins, women_margins,
                                 sigma_x, tau_y, tol=tol, gr=True,
                                 maxiter=maxiter, verbose=verbose)
        # each element of dmus_hxy has the derivatives wrt (Phi, sigma_x, tau_y) in that order
        dmus_xy, dmus_x0, dmus_0y = dmus_hxy.unpack()
        n_prod_categories = ncat_men * ncat_women

        # we need the derivatives wrt Phi and tau
        itau_y = n_prod_categories + ncat_men

        def _reshape_dmu(dmu, n_elems):
            dmur = np.zeros((n_elems, n_prod_categories + 1))
            dmur[:, :n_prod_categories] = dmu[:, :n_prod_categories]
            dmur[:, -1] = np.sum(dmu[:, itau_y:], 1)
            return dmur

        dmuxy = _reshape_dmu(dmus_xy, n_prod_categories)
        dmux0 = _reshape_dmu(dmus_x0, ncat_men)
        dmu0y = _reshape_dmu(dmus_0y, ncat_women)

        return mus, marg_err_x, marg_err_y, MatchingMus(dmuxy, dmux0, dmu0y)


def ipfp_heteroxy_solver(Phi: np.ndarray, men_margins: np.ndarray, women_margins: np.ndarray,
                         sigma_x: np.ndarray, tau_y: np.ndarray, tol: Optional[float] = 1e-9,
                         gr: Optional[bool] = False, verbose: Optional[bool] = False,
                         maxiter: Optional[int] = 1000) -> IPFPReturn:
    """
    solve for equilibrium in a  in a gender- and type-heteroskedastic Choo and Siow market

    given systematic surplus and margins and a scale parameter dist_params[0]

    :param np.array Phi: matrix of systematic surplus, shape (ncat_men, ncat_women)

    :param np.array men_margins: vector of men margins, shape (ncat_men)

    :param np.array women_margins: vector of women margins, shape (ncat_women)

    :param np.array sigma_x: an array of positive numbers of shape (ncat_men)

    :param np.array tau_y: an array of positive numbers of shape (ncat_women)

    :param float tol: tolerance on change in solution

    :param boolean gr: if True, also evaluate derivatives of muxy wrt Phi

    :param boolean verbose: prints stuff

    :param int maxiter: maximum number of iterations

    :return: MatchingMus (muxy, mux0, mu0y); errors on margins marg_err_x, marg_err_y;
             and gradients of (muxy, mux0, mu0y)
             wrt (Phi, sigma_x, tau_y) if gr=True
    """
    if np.min(sigma_x) <= 0.0:
        bs_error_abort("all elements of sigma_x must be positive")
    if np.min(tau_y) <= 0.0:
        bs_error_abort("all elements of tau_y must be positive")

    ncat_men, ncat_women = men_margins.size, women_margins.size
    _test_size_Phi(Phi, ncat_men, ncat_women)

    sumxy1 = 1.0 / np.add.outer(sigma_x, tau_y)
    ephi2 = npexp(Phi * sumxy1)

    rat_sigma = sumxy1 * sigma_x.reshape((-1, 1))
    rat_tau = 1.0 - rat_sigma

    #############################################################################
    # we solve the equilibrium equations muxy = ephi2 * ((mux0^sigma_x) * (mu0y^tau_y))^(1/(sigma_x + tau_y))
    #   with ephi2 = exp(Phi/(sigma_x+tau_y))
    #   starting with a reasonable initial point for mux0 and mu0y: mux0 = mu0y = bigc
    #   bigc chosen to fit the total number of individuals
    #   we work with tx = log(mux0) and ty = log(mu0y) to circumvent positivity constraints
    #############################################################################
    nindivs = np.sum(men_margins) + np.sum(women_margins)
    bigc = nindivs / (ncat_men + ncat_women + 2.0 * np.sum(ephi2))
    lbigc = log(bigc)

    txi = np.full(ncat_men, lbigc)
    tyi = np.full(ncat_women, lbigc)
    err_diff = bigc
    tol_diff = tol * bigc
    tol_newton = tol
    step_newton = 1.0
    niter = 0
    while (err_diff > tol_diff) and (niter < maxiter):
        # Newton iterates for men
        err_newton_x = bigc
        niter_n = 0
        tyim = nprepeat_row(tyi, ncat_men)
        while (err_newton_x > tol_newton) and (niter_n < maxiter):
            txim = nprepeat_col(txi, ncat_women)
            a = npexp(tyim * rat_tau) * ephi2
            muxyi = npexp(txim * rat_sigma) * a
            mux0i = npexp(txi)
            err_x = mux0i + np.sum(muxyi, 1) - men_margins
            derx0 = der_npexp(txim * rat_sigma) * rat_sigma
            der_err_x = der_npexp(txi) + np.sum(derx0 * a, 1)
            err_newton_x = npmaxabs(err_x)
            txi -= step_newton * err_x / der_err_x
            niter_n += 1
        # print(f"Final err_x: {err_newton_x} in {niter_n} iterations")
        # Newton iterates for women
        err_newton_y = bigc
        niter_n = 0
        txim = nprepeat_col(txi, ncat_women)
        while (err_newton_y > tol_newton) and (niter_n < maxiter):
            tyim = nprepeat_row(tyi, ncat_men)
            b = npexp(txim * rat_sigma) * ephi2
            muxyi = npexp(tyim * rat_tau) * b
            mu0yi = npexp(tyi)
            err_y = mu0yi + np.sum(muxyi, 0) - women_margins
            der0y = der_npexp(tyim * rat_tau) * rat_tau
            der_err_y = der_npexp(tyi) + np.sum(der0y * b, 0)
            err_newton_y = npmaxabs(err_y)
            tyi -= step_newton * err_y / der_err_y
            niter_n += 1
        # print(f"Final err_y: {err_newton_y} in {niter_n} iterations")

        # recompute error on men margins
        err_x = mux0i + np.sum(muxyi, 1) - men_margins
        err_diff = npmaxabs(err_x) + err_newton_y

        niter += 1

    # print(f"Final err_diff: {err_diff}")

    mux0 = mux0i
    mu0y = mu0yi
    muxy = muxyi
    marg_err_x = mux0 + np.sum(muxy, 1) - men_margins
    marg_err_y = mu0y + np.sum(muxy, 0) - women_margins
    if verbose:
        print(f"IPFP done after {niter} iterations:")
        print(f"\tMargin error on x: {npmaxabs(marg_err_x)}")
        print(f"\tMargin error on y: {npmaxabs(marg_err_y)}")

    mus = MatchingMus(muxy, mux0, mu0y)

    if not gr:
        return mus, marg_err_x, marg_err_y
    else:  # we compute the derivatives wrt Phi, sigma_x, tau_y
        n_sum_categories = ncat_men + ncat_women
        n_prod_categories = ncat_men * ncat_women

        # some useful terms
        derx0_r = der_npexp(txim * rat_sigma) * txim
        der0y_R = der_npexp(tyim * rat_tau) * tyim
        P_P1_E = b * der0y
        P1_P_E = a * derx0
        delta_term = derx0_r * a - der0y_R * b
        #  this is 1 made safe
        der_logephi2 = der_npexp(Phi * sumxy1) / ephi2

        # start with the LHS of the linear system on (dmux0, dmu0y)
        lhs = np.zeros((n_sum_categories, n_sum_categories))
        lhs[:ncat_men, :ncat_men] = np.diag(der_err_x)
        lhs[:ncat_men, ncat_men:] = P_P1_E
        lhs[ncat_men:, ncat_men:] = np.diag(der_err_y)
        lhs[ncat_men:, :ncat_men] = P1_P_E.T

        # now fill the RHS (derivatives wrt Phi, then sigma_x and tau_y)
        n_cols_rhs = n_prod_categories + n_sum_categories
        rhs = np.zeros((n_sum_categories, n_cols_rhs))
        #  derivatives wrt Phi
        mu_sum = muxy * sumxy1 * der_logephi2
        ivar = 0
        for iman in range(ncat_men):
            rhs[iman, ivar:(ivar + ncat_women)] = -mu_sum[iman, :]
            ivar += ncat_women
        ivar = ncat_men
        for iwoman in range(ncat_women):
            rhs[ivar, iwoman:n_prod_categories:ncat_women] = -mu_sum[:, iwoman]
            ivar += 1
        #  derivatives wrt sigma_x
        mu_sig = (Phi * mu_sum - rat_tau * delta_term) * sumxy1
        for iman in range(ncat_men):
            rhs[iman, n_prod_categories + iman] = np.sum(mu_sig[iman, :])
        ivar = ncat_men
        end_sig = n_prod_categories + ncat_men
        for iwoman in range(ncat_women):
            rhs[ivar, n_prod_categories:end_sig] = mu_sig[:, iwoman]
            ivar += 1
        #  derivatives wrt tau_y
        mu_tau = (Phi * mu_sum + rat_sigma * delta_term) * sumxy1
        ivar = end_sig
        for iman in range(ncat_men):
            rhs[iman, ivar:] = mu_tau[iman, :]
        ivar1 = ncat_men
        ivar2 = end_sig
        for iwoman in range(ncat_women):
            rhs[ivar1, ivar2] = np.sum(mu_tau[:, iwoman])
            ivar1 += 1
            ivar2 += 1

        # solve for the derivatives of tx and ty
        dt = spla.solve(lhs, rhs)
        dtx = dt[:ncat_men, :]
        dty = dt[ncat_men:, :]

        # now fill in dmux0, dmu0y, dmuxy
        dmux0 = dtx * der_npexp(txi).reshape((-1, 1))
        dmu0y = dty * der_npexp(tyi).reshape((-1, 1))
        dmuxy = np.zeros((n_prod_categories, n_cols_rhs))
        ivarx = n_prod_categories
        ivar = 0
        for iman in range(ncat_men):
            ivary = end_sig
            for iwoman in range(ncat_women):
                dmuxy[ivar, :] = \
                    (P1_P_E[iman, iwoman] * dtx[iman, :] +
                     P_P1_E[iman, iwoman] * dty[iwoman, :])
                dmuxy[ivar, ivarx] -= mu_sig[iman, iwoman]
                dmuxy[ivar, ivary] -= mu_tau[iman, iwoman]
                dmuxy[ivar, ivar] += mu_sum[iman, iwoman]
                ivary += 1
                ivar += 1
            ivarx += 1

    return mus, marg_err_x, marg_err_y, MatchingMus(dmuxy, dmux0, dmu0y)


def _print_simulated_ipfp(muxy: np.ndarray, marg_err_x: np.ndarray, marg_err_y: np.ndarray):
    """ prints some information for debugging """
    print("    simulated matching:")
    print(muxy[:4, :4])
    print(f"margin error on x: {npmaxabs(marg_err_x)}")
    print(f"             on y: {npmaxabs(marg_err_y)}")


if __name__ == "__main__":

    do_test_gradient_hetero = True
    do_test_gradient_heteroxy = False

    # we generate a Choo and Siow homoskedastic matching
    ncat_men = ncat_women = 25
    n_sum_categories = ncat_men + ncat_women
    n_prod_categories = ncat_men * ncat_women

    mu, sigma = 0.0, 1.0
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
            bases_surplus[ix, iy, 3] = \
                (x_men[ix] - y_women[iy]) * (x_men[ix] - y_women[iy])

    men_margins = np.random.uniform(1.0, 10.0, size=ncat_men)
    women_margins = np.random.uniform(1.0, 10.0, size=ncat_women)

    # np.random.normal(mu, sigma, size=n_bases)
    true_surplus_params = np.array([3.0, -1.0, -1.0, -2.0])
    true_surplus_matrix = bases_surplus @ true_surplus_params

    print_stars("Testing ipfp homo:")
    mus, marg_err_x, marg_err_y = \
        ipfp_homo_solver(true_surplus_matrix, men_margins,
                         women_margins, tol=1e-12)
    muxy, mux0, mu0y = mus.unpack()
    print("    checking matching:")
    print(" true matching:")
    print(muxy[:4, :4])
    _print_simulated_ipfp(muxy, marg_err_x, marg_err_y)

    # and we test ipfp hetero for tau = 1
    tau = 1.0
    print_stars("Testing ipfp hetero for tau = 1:")
    mus_tau, marg_err_x_tau, marg_err_y_tau = \
        ipfp_hetero_solver(true_surplus_matrix, men_margins,
                           women_margins, tau, verbose=True)
    print("    checking matching:")
    print(" true matching:")
    print(muxy[:4, :4])
    muxy_tau = mus_tau.muxy
    _print_simulated_ipfp(muxy_tau, marg_err_x_tau, marg_err_y_tau)

    # and we test ipfp heteroxy for sigma = tau = 1
    print_stars("Testing ipfp heteroxy for sigma_x and tau_y = 1:")

    sigma_x = np.ones(ncat_men)
    tau_y = np.ones(ncat_women)

    mus_hxy, marg_err_x_hxy, marg_err_y_hxy = \
        ipfp_heteroxy_solver(true_surplus_matrix, men_margins, women_margins,
                             sigma_x, tau_y)
    muxy_hxy = mus_hxy.muxy
    _print_simulated_ipfp(muxy_hxy, marg_err_x_hxy, marg_err_y_hxy)

    # check the grad_f
    iman = 7
    iwoman = 11

    GRADIENT_STEP = 1e-6

    if do_test_gradient_heteroxy:
        mus_hxy, marg_err_x_hxy, marg_err_y_hxy, dmus_hxy = \
            ipfp_heteroxy_solver(true_surplus_matrix, men_margins, women_margins,
                                 sigma_x, tau_y, gr=True)
        muxy_hxy, mux0_hxy, mu0y_hxy = mus_hxy.unpack()
        dmuxy_hxy, dmux0_hxy, dmu0y_hxy = dmus_hxy.unpack()
        muij = muxy_hxy[iman, iwoman]
        muij_x0 = mux0_hxy[iman]
        muij_0y = mu0y_hxy[iwoman]
        gradij = dmuxy_hxy[iman * ncat_women + iwoman, :]
        gradij_x0 = dmux0_hxy[iman, :]
        gradij_0y = dmu0y_hxy[iwoman, :]
        n_cols_rhs = n_prod_categories + n_sum_categories
        gradij_numeric = np.zeros(n_cols_rhs)
        gradij_numeric_x0 = np.zeros(n_cols_rhs)
        gradij_numeric_0y = np.zeros(n_cols_rhs)
        icoef = 0
        for i1 in range(ncat_men):
            for i2 in range(ncat_women):
                surplus_mat = true_surplus_matrix.copy()
                surplus_mat[i1, i2] += GRADIENT_STEP
                mus, marg_err_x, marg_err_y = \
                    ipfp_heteroxy_solver(surplus_mat, men_margins, women_margins,
                                         sigma_x, tau_y)
                muxy, mux0, mu0y = mus.unpack()
                gradij_numeric[icoef] = (
                    muxy[iman, iwoman] - muij) / GRADIENT_STEP
                gradij_numeric_x0[icoef] = (
                    mux0[iman] - muij_x0) / GRADIENT_STEP
                gradij_numeric_0y[icoef] = (
                    mu0y[iwoman] - muij_0y) / GRADIENT_STEP
                icoef += 1
        for ix in range(ncat_men):
            sigma = sigma_x.copy()
            sigma[ix] += GRADIENT_STEP
            mus, marg_err_x, marg_err_y = \
                ipfp_heteroxy_solver(true_surplus_matrix, men_margins, women_margins,
                                     sigma, tau_y)
            muxy, mux0, mu0y = mus.unpack()
            gradij_numeric[icoef] = (muxy[iman, iwoman] - muij) / GRADIENT_STEP
            gradij_numeric_x0[icoef] = (mux0[iman] - muij_x0) / GRADIENT_STEP
            gradij_numeric_0y[icoef] = (mu0y[iwoman] - muij_0y) / GRADIENT_STEP
            icoef += 1
        for iy in range(ncat_women):
            tau = tau_y.copy()
            tau[iy] += GRADIENT_STEP
            mus, marg_err_x, marg_err_y = \
                ipfp_heteroxy_solver(true_surplus_matrix, men_margins, women_margins,
                                     sigma_x, tau)
            muxy, mux0, mu0y = mus.unpack()
            gradij_numeric[icoef] = (muxy[iman, iwoman] - muij) / GRADIENT_STEP
            gradij_numeric_x0[icoef] = (mux0[iman] - muij_x0) / GRADIENT_STEP
            gradij_numeric_0y[icoef] = (mu0y[iwoman] - muij_0y) / GRADIENT_STEP
            icoef += 1

        diff_gradients = gradij_numeric - gradij
        error_gradient = np.abs(diff_gradients)

        describe_array(
            error_gradient, "error on the numerical grad_f, heteroxy")

        diff_gradients_x0 = gradij_numeric_x0 - gradij_x0
        error_gradient_x0 = np.abs(diff_gradients_x0)

        describe_array(error_gradient_x0,
                       "error on the numerical grad_f x0, heteroxy")

        diff_gradients_0y = gradij_numeric_0y - gradij_0y
        error_gradient_0y = np.abs(diff_gradients_0y)

        describe_array(error_gradient_0y,
                       "error on the numerical grad_f 0y, heteroxy")

    if do_test_gradient_hetero:
        tau = 1.0
        mus_h, marg_err_x_h, marg_err_y_h, dmus_h = \
            ipfp_hetero_solver(true_surplus_matrix, men_margins, women_margins,
                               tau, gr=True)
        muxy_h, mux0_h, mu0y_h = mus_h.unpack()
        dmuxy_h, dmux0_h, dmu0y_h = dmus_h.unpack()
        muij = muxy_h[iman, iwoman]
        gradij = dmuxy_h[iman * ncat_women + iwoman, :]
        n_cols_rhs = n_prod_categories + 1
        gradij_numeric = np.zeros(n_cols_rhs)
        icoef = 0
        for i1 in range(ncat_men):
            for i2 in range(ncat_women):
                surplus_mat = true_surplus_matrix.copy()
                surplus_mat[i1, i2] += GRADIENT_STEP
                mus, marg_err_x, marg_err_y = \
                    ipfp_hetero_solver(
                        surplus_mat, men_margins, women_margins, tau)
                muxy, mux0, mu0y = mus.unpack()
                gradij_numeric[icoef] = (
                    muxy[iman, iwoman] - muij) / GRADIENT_STEP
                icoef += 1
        tau_plus = tau + GRADIENT_STEP
        mus, marg_err_x, marg_err_y = \
            ipfp_hetero_solver(true_surplus_matrix,
                               men_margins, women_margins, tau_plus)
        muxy, mux0, mu0y = mus.muxy, mus.mux0, mus.mu0y
        gradij_numeric[-1] = (muxy[iman, iwoman] - muij) / GRADIENT_STEP

        error_gradient = np.abs(gradij_numeric - gradij)

        describe_array(error_gradient, "error on the numerical grad_f, hetero")
