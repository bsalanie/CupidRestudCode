""" read data files and base functions."""

import numpy as np

from cupid_classes import MatchingMus


def read_inputs(datadir):
    muxy_hat = np.loadtxt(datadir + "muxy70nN.txt")

    nx = np.loadtxt(datadir + "nx70n.txt")
    ncat_men = nx.size
    my = np.loadtxt(datadir + "my70n.txt")
    ncat_women = my.size
    n_prod_categories = ncat_men * ncat_women
    mux0_hat = nx - np.sum(muxy_hat, 1)
    mu0y_hat = my - np.sum(muxy_hat, 0)

    # variance-covariance of the mus
    varbigmus_hat = np.loadtxt(datadir + "varmus70nN.txt")
    # the matrix has muxy row major, then  mux0, then mu0y packed in each dimension
    varmus_xyzt = varbigmus_hat[:n_prod_categories, :n_prod_categories]
    varmus_xyz0 = varbigmus_hat[:n_prod_categories, n_prod_categories:(n_prod_categories + ncat_men)]
    varmus_xy0t = varbigmus_hat[:n_prod_categories, (n_prod_categories + ncat_men):]
    varmus_x0z0 = varbigmus_hat[n_prod_categories:(n_prod_categories + ncat_men),
                  n_prod_categories:(n_prod_categories + ncat_men)]
    varmus_x00y = varbigmus_hat[n_prod_categories:(n_prod_categories + ncat_men), (n_prod_categories + ncat_men):]
    varmus_0y0t = varbigmus_hat[(n_prod_categories + ncat_men):, (n_prod_categories + ncat_men):]
    varmus = (varmus_xyzt, varmus_xyz0, varmus_xy0t, varmus_x0z0, varmus_x00y, varmus_0y0t)

    # normalize by the number of households in the population
    n_households_pop = np.sum(nx) + np.sum(my) - np.sum(muxy_hat)

    muxy_hat_norm = muxy_hat / n_households_pop
    mux0_hat_norm = mux0_hat / n_households_pop
    mu0y_hat_norm = mu0y_hat / n_households_pop
    mu_hat_norm = MatchingMus(muxy_hat_norm, mux0_hat_norm, mu0y_hat_norm)
    nx_norm = nx / n_households_pop
    my_norm = my / n_households_pop
    N2 = n_households_pop * n_households_pop
    varmus = tuple(v/N2 for v in varmus)

    # now the bases
    phibases24 = np.loadtxt(datadir + "phibases24.txt")
    n_bases = phibases24.shape[1]
    phibases = np.empty((ncat_men, ncat_women, n_bases))
    i = 0
    for i_base in range(n_bases):
        phibases[:, :, i_base] = \
            phibases24[:, i_base].reshape((ncat_men, ncat_women))

    return mu_hat_norm, nx_norm, my_norm, phibases, varmus
