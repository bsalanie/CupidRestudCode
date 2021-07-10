""" read data files and base functions."""

import numpy as np

from cupid_classes import MatchingMus
from cupid_utils import N_HOUSEHOLDS_POP


def read_inputs(datadir):
    muxy_hat = np.loadtxt(datadir + "muxyhat70nN.txt")

    nx = np.loadtxt(datadir + "nx70n.txt")
    ncat_men = nx.size
    my = np.loadtxt(datadir + "my70n.txt")
    ncat_women = my.size
    n_prod_categories = ncat_men * ncat_women
    mux0_hat = nx - np.sum(muxy_hat, 1)
    mu0y_hat = my - np.sum(muxy_hat, 0)

    # variance-covariance of the mus
    varbigmus_hat = np.loadtxt(datadir + "varbigmu_muxyhat70nN.txt")
    # the matrix has muxy row major, then  mux0, then mu0y packed in each dimension
    varmus_xyzt = varbigmus_hat[:n_prod_categories, :n_prod_categories]
    varmus_xyz0 = varbigmus_hat[:n_prod_categories, n_prod_categories:(n_prod_categories + ncat_men)]
    varmus_xy0t = varbigmus_hat[:n_prod_categories, (n_prod_categories + ncat_men):]
    varmus_x0z0 = varbigmus_hat[n_prod_categories:(n_prod_categories + ncat_men),
                  n_prod_categories:(n_prod_categories + ncat_men)]
    varmus_x00y = varbigmus_hat[n_prod_categories:(n_prod_categories + ncat_men), (n_prod_categories + ncat_men):]
    varmus_0y0t = varbigmus_hat[(n_prod_categories + ncat_men):, (n_prod_categories + ncat_men):]
    varmus = (varmus_xyzt, varmus_xyz0, varmus_xy0t, varmus_x0z0, varmus_x00y, varmus_0y0t)

    # sums of squared weights in each cell
    sumw2_xy = np.loadtxt(datadir + "sumw2_muxyhat70nN.txt")
    sumw2_x0 = np.loadtxt(datadir + "sumw2_mux0hat70nN.txt")
    sumw2_0y = np.loadtxt(datadir + "sumw2_mu0yhat70nN.txt")
    sumw2 = MatchingMus(sumw2_xy, sumw2_x0, sumw2_0y)

    # normalize by the number of households
    muxy_hat_norm = muxy_hat / N_HOUSEHOLDS_POP
    mux0_hat_norm = mux0_hat / N_HOUSEHOLDS_POP
    mu0y_hat_norm = mu0y_hat / N_HOUSEHOLDS_POP
    mu_hat_norm = MatchingMus(muxy_hat_norm, mux0_hat_norm, mu0y_hat_norm)
    nx_norm = nx / N_HOUSEHOLDS_POP
    my_norm = my / N_HOUSEHOLDS_POP

    # now the bases
    phibases2 = np.loadtxt(datadir + "phibases2424.txt")
    n_bases = phibases2.shape[0] // ncat_men
    phibases = np.empty((ncat_men, ncat_women, n_bases))
    i = 0
    for i_base in range(n_bases):
        phibases[:, :, i_base] = \
            phibases2[i: (i + ncat_men), :]
        i += ncat_men

    return mu_hat_norm, nx_norm, my_norm, sumw2, phibases, varmus
