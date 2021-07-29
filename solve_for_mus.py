"""
solves for the stable matching  (muxy, mux0, mu0y) = mus(params, args) and their gradients \
where params = (dist_params, surplus_params)
"""

from typing import Optional, Tuple, Union
import numpy as np
import numpy.linalg as npla

from cupid_classes import MatchingMus, CupidParams, CupidParamsFcmnl, CupidParamsCSHeteroxy
from cupid_utils import bs_error_abort, print_stars, describe_array, GRADIENT_STEP, MIN_MUS_NORM
from cupid_numpy_utils import npexp, der_npexp
from cupid_optim_utils import acc_grad_descent
from ipfp_solvers import ipfp_homo_solver, \
    ipfp_hetero_solver, ipfp_heteroxy_solver
from fcmnl import grad_GplusH_fcmnl, \
    make_b1, make_b2, make_b3, make_b4, make_b5, make_b8, derivs_GplusH_fcmnl
from solve_fcmnl import minimize_fcmnl_U

# this is the type each mus_XXX function returns
MusReturn = Union[Tuple[MatchingMus, np.ndarray,
                        MatchingMus], Tuple[MatchingMus, np.ndarray]]


def mus_choosiow_and_maybe_grad(params: np.ndarray, model_params: CupidParams,
                                gr: Optional[bool] = False) -> MusReturn:
    """
    computes muxy, mux0, mu0y and their gradients if gr for the homoskedastic CS model

    :param np.ndarray params: coefficients of bases

    :param CupidParams model_params: parameters of model

    :param boolean gr: if True, we compute the  gradient

    :return: MatchingMus and Uconv, and maybe the gradient of the mus wrt the params
    """

    bases_surplus = model_params.bases_surplus
    men_margins = model_params.men_margins
    women_margins = model_params.women_margins

    ncat_men, ncat_women, n_bases = bases_surplus.shape
    surplus_params_vals = params
    Phi = bases_surplus @ surplus_params_vals

    n_params = params.size

    if gr:
        mus, _, _, dmus = ipfp_homo_solver(
            Phi, men_margins, women_margins, gr=True)
        U_conv = np.log(mus.muxy / mus.mux0.reshape((-1, 1)))
        dmuxy_Phi = dmus.muxy
        dmuxy = np.empty((ncat_men, ncat_women, n_bases))
        # derivatives wrt coefficients of bases
        n_prod_categories = ncat_men * ncat_women
        for ibase in range(n_bases):
            base_i = bases_surplus[:, :, ibase].reshape(n_prod_categories)
            dmuxy[:, :, ibase] = (
                dmuxy_Phi @ base_i).reshape((ncat_men, ncat_women))

        dmux0 = np.empty((ncat_men, n_params))
        dmu0y = np.empty((ncat_women, n_params))
        for iman in range(ncat_men):
            dmux0[iman, :] = -np.sum(dmuxy[iman, :, :], 0)
        for iwoman in range(ncat_women):
            dmu0y[iwoman, :] = -np.sum(dmuxy[:, iwoman, :], 0)
        dmus_params = MatchingMus(dmuxy, dmux0, dmu0y)
        return mus, U_conv, dmus_params
    else:
        mus, _, _ = ipfp_homo_solver(Phi, men_margins, women_margins, gr=False)
        U_conv = np.log(mus.muxy / mus.mux0.reshape((-1, 1)))
        return mus, U_conv


def mus_choosiow_hetero_and_maybe_grad(params: np.ndarray, model_params: CupidParams,
                                       gr: Optional[bool] = False) -> MusReturn:
    """
    computes muxy, mux0, mu0y and their gradients if gr for the gender-heteroskedastic CS model

    :param np.ndarray params: tau and coefficients of bases

    :param CupidParams model_params: parameters of model

    :param boolean gr: if True, we compute the  gradient

    :return: MatchingMus and Uconv, and maybe the gradient of the mus wrt the params
    """
    bases_surplus = model_params.bases_surplus
    men_margins = model_params.men_margins
    women_margins = model_params.women_margins

    ncat_men, ncat_women, n_bases = bases_surplus.shape
    n_params = params.size

    tau = params[0]
    surplus_params_vals = params[1:]
    Phi = bases_surplus @ surplus_params_vals

    if gr:
        mus, _, _, dmus = ipfp_hetero_solver(Phi, men_margins, women_margins, tau,
                                             gr=True)
        U_conv = np.log(mus.muxy / mus.mux0.reshape((-1, 1)))
        dmuxy_Phi = dmus.muxy[:, :-1]
        dmuxy = np.empty((ncat_men, ncat_women, n_params))
        # derivatives wrt tau
        dmuxy[:, :, 0] = dmus.muxy[:, -1].reshape((ncat_men, ncat_women))
        # derivatives wrt coefficients of bases
        n_prod_categories = ncat_men * ncat_women
        for ibase in range(n_bases):
            base_i = bases_surplus[:, :, ibase].reshape(n_prod_categories)
            dmuxy[:, :, 1 + ibase] = (dmuxy_Phi @
                                      base_i).reshape((ncat_men, ncat_women))

        dmux0 = np.empty((ncat_men, n_params))
        dmu0y = np.empty((ncat_women, n_params))
        for iman in range(ncat_men):
            dmux0[iman, :] = -np.sum(dmuxy[iman, :, :], 0)
        for iwoman in range(ncat_women):
            dmu0y[iwoman, :] = -np.sum(dmuxy[:, iwoman, :], 0)

        dmus_params = MatchingMus(dmuxy, dmux0, dmu0y)
        return mus, U_conv, dmus_params
    else:
        mus, _, _ = \
            ipfp_hetero_solver(Phi, men_margins, women_margins, tau, gr=False)
        U_conv = np.log(mus.muxy / mus.mux0.reshape((-1, 1)))
        return mus, U_conv


def mus_choosiow_heteroxy_and_maybe_grad(params: np.ndarray, model_params: CupidParamsCSHeteroxy,
                                         gr: Optional[bool] = False) -> MusReturn:
    """
    computes muxy, mux0, mu0y and their gradients if gr for the gender- and age-heteroskedastic CS model

    :param np.ndarray params: parameters of sigma_x and tau_y and coefficients of bases

    :param CupidParamsCSHeteroxy model_params: parameters of model

    :param boolean gr: if True, we compute the  gradient

    :return: MatchingMus and Uconv, and maybe the gradient of the mus wrt the params
    """

    bases_surplus = model_params.bases_surplus
    men_margins = model_params.men_margins
    women_margins = model_params.women_margins
    covariates_sigma = model_params.covariates_sigma
    covariates_tau = model_params.covariates_tau

    n_sigma = covariates_sigma.shape[1]
    n_tau = covariates_tau.shape[1]
    n_params = params.size

    ncat_men, ncat_women, n_bases = bases_surplus.shape

    n_dist_params = n_sigma + n_tau
    sigma_pars = params[:n_sigma]
    X_sigma = covariates_sigma @ sigma_pars
    sigma_x = npexp(X_sigma)
    dsigma_x = covariates_sigma * der_npexp(X_sigma).reshape((-1, 1))
    tau_pars = params[n_sigma:n_dist_params]
    X_tau = covariates_tau @ tau_pars
    tau_y = npexp(X_tau)
    dtau_y = covariates_tau * der_npexp(X_tau).reshape((-1, 1))

    surplus_params_vals = params[n_dist_params:]
    Phi = bases_surplus @ surplus_params_vals

    if gr:
        mus, _, _, dmus = ipfp_heteroxy_solver(
            Phi, men_margins, women_margins, sigma_x, tau_y, gr=True)
        U_conv = np.log(mus.muxy / mus.mux0.reshape((-1, 1))) / \
            sigma_x.reshape((-1, 1))
        # we have the derivatives wrt (PHi, sigma_x, tau_y)
        #  we want the derivatives wrt params
        n_prod_categories = ncat_men * ncat_women
        end_sig = n_prod_categories + ncat_men
        dmus_xy = dmus.muxy

        dmuxy = np.zeros((ncat_men, ncat_women, n_params))

        # derivatives wrt the coefficients of sigma_x
        dmus_xy_sig = dmus_xy[:, n_prod_categories:end_sig]
        dmuxy_sig = dmus_xy_sig @ dsigma_x
        for isig in range(n_sigma):
            dmuxy[:, :, isig] = dmuxy_sig[:, isig].reshape(
                (ncat_men, ncat_women))
        # derivatives wrt the coefficients of tau_y
        dmus_xy_tau = dmus_xy[:, end_sig:]
        dmuxy_tau = dmus_xy_tau @ dtau_y
        for itau in range(n_tau):
            dmuxy[:, :, n_sigma + itau] = dmuxy_tau[:,
                                                    itau].reshape((ncat_men, ncat_women))
        # derivatives wrt the coefficients of the bases
        dmus_xy_Phi = dmus_xy[:, :n_prod_categories]
        ivar = n_dist_params
        for ibase in range(n_bases):
            base_i = bases_surplus[:, :, ibase].reshape(n_prod_categories)
            dmuxy[:, :, ivar] = (
                dmus_xy_Phi @ base_i).reshape((ncat_men, ncat_women))
            ivar += 1

        dmux0 = np.empty((ncat_men, n_params))
        dmu0y = np.empty((ncat_women, n_params))
        for iman in range(ncat_men):
            dmux0[iman, :] = -np.sum(dmuxy[iman, :, :], 0)
        for iwoman in range(ncat_women):
            dmu0y[iwoman, :] = -np.sum(dmuxy[:, iwoman, :], 0)

        dmus_params = MatchingMus(dmuxy, dmux0, dmu0y)

        return mus, U_conv, dmus_params
    else:
        mus, _, _ = ipfp_heteroxy_solver(
            Phi, men_margins, women_margins, sigma_x, tau_y, gr=False)
        U_conv = np.log(mus.muxy / mus.mux0.reshape((-1, 1))) / \
            sigma_x.reshape((-1, 1))
        return mus, U_conv


def mus_fcmnl_and_maybe_grad_agd(params: np.ndarray, model_params: CupidParamsFcmnl,
                                 gr: Optional[bool] = False, verbose: Optional[bool] = False) -> MusReturn:
    """
    computes muxy, mux0, mu0y and their gradients if gr for the FC-MNL model
    using accelerated gradient descent

    :param np.ndarray params: parameters of b and coefficients of bases

    :param CupidParamsFcmnl model_params: parameters of model

    :param boolean gr: if True, we compute the  gradient

    :param boolean verbose: if True, we print stuff

    :return: MatchingMus and Uconv, and maybe the gradient of the mus wrt the params
    """

    bases_surplus = model_params.bases_surplus
    men_margins, women_margins = model_params.men_margins, model_params.women_margins
    npars_b_men, npars_b_women = model_params.n_pars_b_men, model_params.n_pars_b_women

    # tolerance for AGD
    tol = model_params.tol_agd

    npars_b = npars_b_men + npars_b_women

    ncat_men, ncat_women, n_bases = bases_surplus.shape
    n_prod_categories = ncat_men * ncat_women

    n_params = params.size

    pars_b_men = params[:npars_b_men]
    pars_b_women = params[npars_b_men:npars_b]
    surplus_params_vals = params[npars_b:]
    Phi = bases_surplus @ surplus_params_vals

    Phiv = Phi.reshape(n_prod_categories)
    U_init = Phiv / 2.0

    mu_hat = model_params.observed_matching
    if mu_hat is not None:
        mu_rat = np.maximum(
            mu_hat.muxy/((mu_hat.mux0).reshape((-1, 1))), MIN_MUS_NORM)
        U_hat_homo = np.log(mu_rat)
        U_init = U_hat_homo.reshape(n_prod_categories)/model_params.tau

    # U_conv, ret_code = \
    #     acc_grad_descent(grad_GplusH_fcmnl, U_init, other_params=(model_params, Phi, pars_b_men, pars_b_women),
    #                      verbose=False, print_result=True, tol=tol)

    U_conv, ret_code = \
        minimize_fcmnl_U(U_init, other_params=(model_params, Phi, pars_b_men, pars_b_women),
                         verbose=False)

    if ret_code == 0:
        if verbose:
            print("mus_fcmnl_and_maybe_grad_agd: AGD converged fine")
        if not gr:
            # mu is the gradient of G wrt U
            resus_ders = derivs_GplusH_fcmnl(U_conv, model_params, Phi, pars_b_men, pars_b_women,
                                             derivs=1)
            muxy = resus_ders.gradients[1]  # this is mu = G_U
            if verbose:
                print(f" gradient  has norm {np.linalg.norm(muxy - resus_ders.gradients[3])}")
            mux0 = men_margins - np.sum(muxy, 1)
            mu0y = women_margins - np.sum(muxy, 0)
            mus = MatchingMus(muxy, mux0, mu0y)
            # print("mus: computed w/o gradient")
            return mus, U_conv
        else:
            # to  compute the gradient of mu, we need the hessians of G and H
            resus_ders = derivs_GplusH_fcmnl(U_conv, model_params, Phi, pars_b_men, pars_b_women,
                                             derivs=2)
            muxy = resus_ders.gradients[1]
            mux0 = men_margins - np.sum(muxy, 1)
            mu0y = women_margins - np.sum(muxy, 0)
            mus = MatchingMus(muxy, mux0, mu0y)

            d2G_dU = resus_ders.hessians[0]
            d2G_UU = resus_ders.hessians[1]
            d2H_dV = resus_ders.hessians[2]
            d2H_VV = resus_ders.hessians[3]

            # we solve (G_UU+H_VV) dU = (H_dV-G_dU) db + H_VV dPhi in dU
            lhs = d2G_UU + d2H_VV
            d2rhs = np.column_stack(((d2H_dV - d2G_dU).T, d2H_VV))
            dU_mat = npla.solve(lhs, d2rhs)
            # and we use dmu = G_UU dU + G_dU db
            dmuxy_mat = d2G_UU @ dU_mat
            dmuxy_mat[:, :npars_b] += d2G_dU.T

            # dmuxy has the derivatives in rows, wrt b then wrt Phi
            # we need the dervatives wrt the parameters
            dmuxy_params = np.empty((ncat_men, ncat_women, n_params))

            # we already have them for b
            for ipar_b in range(npars_b):
                dmuxy_params[:, :, ipar_b] = dmuxy_mat[:,
                                                       ipar_b].reshape((ncat_men, ncat_women))
            # we need to transform them for the coefficients of the bases
            dmuxy_Phi = dmuxy_mat[:, npars_b:]

            i = npars_b
            for ibase in range(n_bases):
                phibase_i = bases_surplus[:, :, ibase].reshape(
                    n_prod_categories)
                dmuxy_params[:, :, i] = (
                    dmuxy_Phi @ phibase_i).reshape((ncat_men, ncat_women))
                i += 1

            dmux0_params = np.empty((ncat_men, n_params))
            dmu0y_params = np.empty((ncat_women, n_params))
            for iman in range(ncat_men):
                dmux0_params[iman, :] = -np.sum(dmuxy_params[iman, :, :], 0)
            for iwoman in range(ncat_women):
                dmu0y_params[iwoman, :] = - \
                    np.sum(dmuxy_params[:, iwoman, :], 0)

            dmus_params = MatchingMus(dmuxy_params, dmux0_params, dmu0y_params)
            # print("mus: computed with gradient")
            return mus, U_conv, dmus_params

    else:
        bs_error_abort(f"AGD did not converge; return code {ret_code}")


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

    men_margins = np.random.uniform(1.0, 10.0, size=ncat_men)
    women_margins = np.random.uniform(1.0, 10.0, size=ncat_women)

    true_surplus_params = np.array([3.0, -1.0, -1.0, -2.0])
    true_surplus_matrix = bases_surplus @ true_surplus_params

    model_params = CupidParams(men_margins=men_margins, women_margins=women_margins,
                               bases_surplus=bases_surplus,
                               observed_matching=None, mus_and_maybe_grad=None)

    EPS = GRADIENT_STEP

    test_homo = True
    test_hetero = True
    test_fcmnl = True

    if test_homo:
        print_stars("Checking the homoskedastic model")
        # test Choo Siow homogeneous
        mus, _, dmus = mus_choosiow_and_maybe_grad(
            true_surplus_params, model_params, gr=True)
        muxy, mux0, mu0y = mus.unpack()
        dmuxy_num = np.empty((ncat_men, ncat_women, n_bases))
        dmux0_num = np.empty((ncat_men, n_bases))
        dmu0y_num = np.empty((ncat_women, n_bases))

        for ibase in range(n_bases):
            surpl_par1 = true_surplus_params.copy()
            surpl_par1[ibase] += EPS
            mus1, _ = mus_choosiow_and_maybe_grad(
                surpl_par1, model_params, gr=False)
            dmuxy_num[:, :, ibase] = (mus1.muxy - muxy) / EPS
            dmux0_num[:, ibase] = (mus1.mux0 - mux0) / EPS
            dmu0y_num[:, ibase] = (mus1.mu0y - mu0y) / EPS

        dmuxy = dmus.muxy
        error_dmuxy = dmuxy_num - dmuxy
        describe_array(error_dmuxy, "error dmuxy")

        dmux0 = dmus.mux0
        error_dmux0 = dmux0_num - dmux0
        describe_array(error_dmux0, "error dmux0")

        dmu0y = dmus.mu0y
        error_dmu0y = dmu0y_num - dmu0y
        describe_array(error_dmu0y, "error dmu0y")

    if test_hetero:
        print_stars("Checking the gender-heteroskedastic model")
        # test Choo Siow tau-heterogeneous

        n_params = 1 + n_bases
        params = np.zeros(n_params)
        tau = 1.3

        params[0] = tau

        mus, _, dmus = mus_choosiow_hetero_and_maybe_grad(
            params, model_params, gr=True)
        muxy, mux0, mu0y = mus.unpack()
        dmuxy_num = np.empty((ncat_men, ncat_women, n_params))
        dmux0_num = np.empty((ncat_men, n_params))
        dmu0y_num = np.empty((ncat_women, n_params))

        for ipar in range(n_params):
            par1 = params.copy()
            par1[ipar] += EPS
            mus1, _ = mus_choosiow_hetero_and_maybe_grad(
                par1, model_params, gr=False)
            dmuxy_num[:, :, ipar] = (mus1.muxy - muxy) / EPS
            dmux0_num[:, ipar] = (mus1.mux0 - mux0) / EPS
            dmu0y_num[:, ipar] = (mus1.mu0y - mu0y) / EPS

        dmuxy = dmus.muxy
        error_dmuxy = dmuxy_num - dmuxy
        describe_array(error_dmuxy, "error dmuxy")

        dmux0 = dmus.mux0
        error_dmux0 = dmux0_num - dmux0
        describe_array(error_dmux0, "error dmux0")

        dmu0y = dmus.mu0y
        error_dmu0y = dmu0y_num - dmu0y
        describe_array(error_dmu0y, "error dmu0y")

    if test_fcmnl:
        # test FCMNL
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
                                        observed_matching=None,
                                        mus_and_maybe_grad=mus_fcmnl_and_maybe_grad_agd,
                                        tol_agd=1e-12)

        n_params = n_pars_b + n_bases
        params = np.zeros(n_params)
        params[:n_pars_b_men] = pars_b_men_arr
        params[n_pars_b_men:n_pars_b] = pars_b_women_arr

        mus, U_conv, dmus = mus_fcmnl_and_maybe_grad_agd(
            params, model_params, gr=True)
        muxy, mux0, mu0y = mus.unpack()
        dmuxy_num = np.empty((ncat_men, ncat_women, n_params))
        dmux0_num = np.empty((ncat_men, n_params))
        dmu0y_num = np.empty((ncat_women, n_params))

        for ipar in range(n_params):
            par1 = params.copy()
            par1[ipar] += EPS
            mus1, _ = mus_fcmnl_and_maybe_grad_agd(
                par1, model_params, gr=False)
            dmuxy_num[:, :, ipar] = (mus1.muxy - muxy) / EPS
            dmux0_num[:, ipar] = (mus1.mux0 - mux0) / EPS
            dmu0y_num[:, ipar] = (mus1.mu0y - mu0y) / EPS
            error_ipar = dmuxy_num[:, :, ipar] - dmus.muxy[:, :, ipar]
            describe_array(error_ipar, f"error dmuxy[{ipar}]")

    dmuxy = dmus.muxy
    error_dmuxy = dmuxy_num - dmuxy
    describe_array(error_dmuxy, "error dmuxy")

    dmux0 = dmus.mux0
    error_dmux0 = dmux0_num - dmux0
    describe_array(error_dmux0, "error dmux0")

    dmu0y = dmus.mu0y
    error_dmu0y = dmu0y_num - dmu0y
    describe_array(error_dmu0y, "error dmu0y")
