"""
estimate variants of separable models on the Choo and Siow data using maximum likelihood on (mux0, mu0y, params)
"""

from math import inf
import numpy as np

from cupid_classes import CupidParams, CupidParamsFcmnl, CupidParamsCSHeteroxy
from cupid_utils import root_dir, print_stars, bs_error_abort, MIN_MUS_NORM

from read_inputs import read_inputs

from ipfp_solvers import ipfp_homo_solver, ipfp_hetero_solver, ipfp_heteroxy_solver

from estimate_model import maximize_loglik, maximize_loglik_fcmnl_MPEC
from analyze_results import analyze_results

from solve_for_mus import mus_choosiow_and_maybe_grad, \
    mus_choosiow_hetero_and_maybe_grad, \
    mus_choosiow_heteroxy_and_maybe_grad, \
    mus_fcmnl_and_maybe_grad_agd

from fcmnl import make_b0, make_b1, make_b2, make_b3, make_b4, \
    make_b5, make_b6, make_b7, make_b8

results_dir = root_dir / "Results"

do_ChooSiow_homoskedastic = False
do_ChooSiow_gender_heteroskedastic = False
do_ChooSiow_gender_age_heteroskedastic = False
do_maxi_fcmnl = True
do_maxi_fcmnl_MPEC = True

# first, read the data
data_dir = root_dir / "Data" / "Output"
mu_hat_norm, nx_norm, my_norm, \
    phibases, varmus = read_inputs(data_dir)

# dimensions
ncat_men, ncat_women, n_bases = phibases.shape

# some initial values for the coefficients of the bases
theta_bases_init = np.zeros(n_bases)
theta_bases_init[0] = -10.0

# set up the Choo-Siow homoskedastic model
dist_params = None
bases_params = theta_bases_init

cs_homo_params_norm = CupidParams(men_margins=nx_norm, women_margins=my_norm,
                                  observed_matching=mu_hat_norm,
                                  bases_surplus=phibases,
                                  mus_and_maybe_grad=mus_choosiow_and_maybe_grad,
                                  ipfp_solver=ipfp_homo_solver)

if do_ChooSiow_homoskedastic:
    print("\n\n" + '*' * 60)
    print("\n\n now we estimate a Choo and Siow homoskedastic model")
    print("\n\n" + '*' * 60)

    x_init = theta_bases_init

    loglik_homo, estimates_surplus_homo, status_homo \
        = maximize_loglik(cs_homo_params_norm, x_init, verbose=False,
                          checkgrad=False)

    print_stars(f"Return status: {status_homo}")

    surplus_params_estimates = estimates_surplus_homo

    analyze_results(cs_homo_params_norm, surplus_params_estimates, 
                    "homoskedastic",
                    results_dir=results_dir,
                    do_stderrs=True, varmus=varmus, save=True)

if do_ChooSiow_gender_heteroskedastic:
    print("\n\n" + '*' * 60)
    print(f"\n\n now we estimate a Choo-Siow gender-heteroskedastic model")
    print("\n\n" + '*' * 60)

    tau_low = 0.5
    tau_high = 1.5

    tau_init_arr = np.array([(tau_low + tau_high) / 2.0])
    dist_params = tau_init_arr

    n_params = 1 + n_bases
    # bounds
    lower = np.full(n_params, -inf)
    lower[0] = tau_low
    upper = np.full(n_params, inf)
    upper[0] = tau_high

    cs_hetero_params_norm = cs_homo_params_norm
    cs_hetero_params_norm.mus_and_maybe_grad = mus_choosiow_hetero_and_maybe_grad
    cs_hetero_params_norm.ipfp_solver = ipfp_hetero_solver
    cs_hetero_params_norm.dist_params = dist_params

    x_init = np.concatenate(
        (tau_init_arr, theta_bases_init))

    loglik_hetero, estimates_hetero, status_hetero \
        = maximize_loglik(cs_hetero_params_norm, x_init,
                          lower=lower, upper=upper, checkgrad=False,
                          verbose=False)

    print_stars(f"Return status: {status_hetero}")

    analyze_results(cs_hetero_params_norm, estimates_hetero,
                    "gender_heteroskedastic",
                    results_dir=results_dir, 
                    do_stderrs=True,
                    varmus=varmus,
                    save=True)

if do_ChooSiow_gender_age_heteroskedastic:

    print("\n\n" + '*' * 60)
    print(f"\n\n now we estimate a Choo and Siow gender- and age-heteroskedastic model")
    print("\n\n" + '*' * 60)

    # use the homoskedastic estimates for the bases
    theta_bases_init = np.loadtxt(results_dir / "homoskedastic" \
                                  / "thetas.txt")

    # select the bases functions which sigma_x and tau_y depend on
    indices_bases_sigma = [10, 20]
    indices_bases_tau = [0]
    str_covariates = ""
    for i in indices_bases_sigma:
        str_covariates += str(i)
    str_covariates += "_"
    for i in indices_bases_tau:
        str_covariates += str(i)
    n_sigma_pars = len(indices_bases_sigma)
    n_tau_pars = len(indices_bases_tau)
    covariates_sigma = np.empty((ncat_men, n_sigma_pars))
    for i in range(n_sigma_pars):
        covariates_sigma[:, i] = phibases[:, 0, indices_bases_sigma[i]]
    covariates_tau = np.empty((ncat_women, n_tau_pars))
    for i in range(n_tau_pars):
        covariates_tau[:, i] = phibases[0, :, indices_bases_tau[i]]
    # initial values for the parameters of sigma_x and tau_y
    sigma_pars_init = np.full(n_sigma_pars, 0.0)
    tau_pars_init = np.full(n_tau_pars, 0.0)
    sigma_tau_pars_init = np.concatenate((sigma_pars_init, tau_pars_init))

    dist_params = sigma_tau_pars_init
    n_dist_params = n_sigma_pars + n_tau_pars
    n_params = n_dist_params + n_bases

    # bounds on sigma_pars and tau_pars
    lower = np.full(n_params, -inf)
    lower[:n_dist_params] = -3.0
    upper = np.full(n_params, inf)
    upper[:n_dist_params] = 3.0

    cs_heteroxy_params_norm = CupidParamsCSHeteroxy(men_margins=nx_norm, women_margins=my_norm,
                                                    observed_matching=mu_hat_norm,
                                                    bases_surplus=phibases,
                                                    mus_and_maybe_grad=mus_choosiow_heteroxy_and_maybe_grad,
                                                    ipfp_solver=ipfp_heteroxy_solver,
                                                    covariates_sigma=covariates_sigma,
                                                    covariates_tau=covariates_tau)

    x_init = np.concatenate((sigma_tau_pars_init, theta_bases_init))
    loglik_heteroxy, estimates_heteroxy, status_heteroxy \
        = maximize_loglik(cs_heteroxy_params_norm, x_init,
                          lower=lower, upper=upper, checkgrad=False,
                          verbose=True)

    print_stars(f"Return status: {status_heteroxy}")

    analyze_results(cs_heteroxy_params_norm, estimates_heteroxy,
                    "gender_age_heteroskedastic_" + str_covariates,
                    results_dir=results_dir,
                    do_stderrs=True,  varmus=varmus, save=True)

if do_maxi_fcmnl:
    for b_case in [5]:

        print("\n\n" + '*' * 60)
        print(f"\n\n now we estimate an FC-MNL model, case {b_case}")
        print("\n\n" + '*' * 60)

        if b_case == 0:                                  # b = Identity
            pars_b_men_init = np.array([])
            pars_b_women_init = np.array([])
            make_b = make_b0
        elif b_case == 1:                                  # orders (1,0)
            pars_b_men_init = np.full(1, 0.1)
            pars_b_women_init = np.array([])
            make_b = make_b1
        elif b_case == 2:                                # orders (0,1)
            pars_b_men_init = np.array([])
            pars_b_women_init = np.full(1, 0.1)
            make_b = make_b2
        elif b_case == 3:                                # orders (2,0)
            pars_b_men_init = np.full(2, 0.1)
            pars_b_women_init = np.array([])
            make_b = make_b3
        elif b_case == 4:                                # orders (1,1)
            pars_b_men_init = np.full(1, 0.1)
            pars_b_women_init = np.full(1, 0.1)
            make_b = make_b4
        elif b_case == 5:                                # orders (0,2) --- the BIC-best model
            pars_b_men_init = np.array([])
            pars_b_women_init = np.array([0.05, 0.02])
            make_b = make_b5
        elif b_case == 6:                                # orders (2,1)
            pars_b_men_init = np.full(2, 0.1)
            pars_b_women_init = np.full(1, 0.1)
            make_b = make_b6
        elif b_case == 7:  # orders (1,2)
            pars_b_men_init = np.full(1, 0.1)
            pars_b_women_init = np.full(2, 0.1)
            make_b = make_b7
        elif b_case == 8:  #  orders (2,2)
            pars_b_men_init = np.full(2, 0.1)
            pars_b_women_init = np.full(2, 0.1)
            make_b = make_b8
        else:
            bs_error_abort(f"No such thing as b_case={b_case}")

        n_pars_b_men = pars_b_men_init.size
        n_pars_b_women = pars_b_women_init.size
        n_pars_b = n_pars_b_men + n_pars_b_women

        pars_b_init = np.concatenate((pars_b_men_init, pars_b_women_init))

        # FCMNL default parameters
        sigma = 0.5
        tau = 1.1


        fcmnl_params_norm = CupidParamsFcmnl(men_margins=nx_norm, women_margins=my_norm,
                                             observed_matching=mu_hat_norm,
                                             bases_surplus=phibases,
                                             mus_and_maybe_grad=mus_fcmnl_and_maybe_grad_agd,
                                             tol_agd=1e-12,
                                             make_b=make_b,
                                             n_pars_b_men=n_pars_b_men,
                                             n_pars_b_women=n_pars_b_women)


        if do_maxi_fcmnl:
            # we read the Choo and Siow homoskedastic estimates of the coefficients of the bases
            # estimates_homo = np.loadtxt(
            #    results_dir / "homoskedastic" / "thetas.txt")
            # they need to be rescaled
            # x_bases_init = estimates_homo/tau

            x_init = np.loadtxt("current_pars.txt")
            x_init = x_init[2:]
            x_init[0] = 0.05
            x_init[1] = 0.02
            # x_init = np.concatenate((pars_b_init, x_bases_init))

            n_params = n_pars_b + n_bases

            # bounds on pars_b_men and pars_b_women
            lower = np.full(n_params, -inf)
            lower[:n_pars_b] = 0.0
            upper = np.full(n_params, inf)
            upper[:n_pars_b] = 0.5

            loglik_fcmnl, estimates_fcmnl, status_fcmnl = maximize_loglik(fcmnl_params_norm, x_init,
                                                                          lower=lower, upper=upper,
                                                                          checkgrad=False,
                                                                          verbose=True)
            print_stars(f"Return status: {status_fcmnl}")

            analyze_results(fcmnl_params_norm, estimates_fcmnl, 
                            "Fcmnl_b" + str(b_case),
                            results_dir=results_dir,
                            save=True)


        if do_maxi_fcmnl_MPEC:

            n_thetas = n_pars_b + n_bases
            n_prod_categories = ncat_men*ncat_women
            n_paramsU = n_thetas + n_prod_categories

            # we read the Choo and Siow homoskedastic estimates of the coefficients of the bases
            estimates_homo = np.loadtxt(
               results_dir / "homoskedastic" / "thetas.txt")
            # they need to be rescaled
            x_bases_init = estimates_homo/tau

            thetas_init = np.zeros(n_thetas)
            thetas_init[0] = 0.01
            thetas_init[1] = 0.01
            thetas_init[2:] = x_bases_init

            mu_hat = fcmnl_params_norm.observed_matching
            if mu_hat is not None:
                mu_rat = np.maximum(
                    mu_hat.muxy / ((mu_hat.mux0).reshape((-1, 1))), MIN_MUS_NORM)
                U_hat_homo = np.log(mu_rat)
                U_init = U_hat_homo.reshape(n_prod_categories) / tau

            x_init = np.concatenate((thetas_init, U_init))

            # bounds on pars_b_men and pars_b_women
            lower = np.full(n_paramsU, -inf)
            lower[:n_pars_b] = 0.0
            upper = np.full(n_paramsU, inf)
            upper[:n_pars_b] = 0.5

            loglik_fcmnl, estimates_fcmnl, status_fcmnl = maximize_loglik_fcmnl_MPEC(fcmnl_params_norm, x_init,
                                                                          lower=lower, upper=upper,
                                                                          checkgrad=True,
                                                                          verbose=True)
            print_stars(f"Return status: {status_fcmnl}")

