"""
Plot the simulated matching patterns for FCMNL
"""

import numpy as np
from math import log

from cupid_utils import root_dir, N_HOUSEHOLDS_OBS, print_stars
from read_inputs import read_inputs
from fcmnl import derivs_GplusH_fcmnl

import matplotlib.pyplot as plt

plt.style.use('ggplot')

results_dir = root_dir + "Results/"
data_dir = root_dir + "Data/"
plots_dir = root_dir + "Plots/"

# first, read the data
mu_hat_norm, nx_norm, my_norm, sumw2, phibases, varmus = read_inputs(data_dir)
muxy_hat_norm, mux0_hat_norm, mu0y_hat_norm = mu_hat_norm.unpack()

# dimensions
ncat_men, ncat_women, n_bases = phibases.shape

# where we stored the results
homo_dir = results_dir + "Homskedastic/"
fcmnl_dir = results_dir + "Fcmnl/"

str_title_homo = "Choo-Siow homoskedastic"
str_title_fcmnl = "Fcmnl"

npars_fcmnl_5 = 32
npars_b_5 = 2

fits = np.loadtxt(fcmnl_dir +"Fcmnl_b5_fits.txt")
# total loglik
fits[0] *= N_HOUSEHOLDS_OBS
# recompute AIC, BIC
fits[1] = fits[0] - npars_fcmnl_5
fits[2] = fits[0] - npars_fcmnl_5 * log(N_HOUSEHOLDS_OBS) / 2.0

# read the simulated mus
muxy_norm_homo = np.loadtxt(homo_dir + "CS_homoskedastic_muxy_norm.txt")
mux0_norm_homo = np.loadtxt(homo_dir + "CS_homoskedastic_mux0_norm.txt")
mu0y_norm_homo = np.loadtxt(homo_dir + "CS_homoskedastic_mu0y_norm.txt")
muxy_norm_fcmnl5 = np.loadtxt(fcmnl_dir + "Fcmnl_b5_muxy_norm.txt")
mux0_norm_fcmnl5 = np.loadtxt(fcmnl_dir + "Fcmnl_b5_mux0_norm.txt")
mu0y_norm_fcmnl5 = np.loadtxt(fcmnl_dir + "Fcmnl_b5_mu0y_norm.txt")


nx_homo = np.sum(muxy_norm_homo, 1) + mux0_norm_homo
my_homo = np.sum(muxy_norm_homo, 0) + mu0y_norm_homo
mugivenx_homo = muxy_norm_homo / nx_homo.reshape((-1, 1))
mugiveny_homo = muxy_norm_homo / my_homo
nx_fcmnl5 = np.sum(muxy_norm_fcmnl5, 1) + mux0_norm_fcmnl5
my_fcmnl5 = np.sum(muxy_norm_fcmnl5, 0) + mu0y_norm_fcmnl5
mugivenx_fcmnl5 = muxy_norm_fcmnl5 / nx_fcmnl5.reshape((-1, 1))
mugiveny_fcmnl5 = muxy_norm_fcmnl5 / my_fcmnl5

d2H_homo = []
for iwoman in range(ncat_women):
    muxy_woman = muxy_norm_homo[:, iwoman]
    mugiveny_woman = mugiveny_homo[:, iwoman]
    d2H_th = np.diag(muxy_woman) - np.outer(muxy_woman, mugiveny_woman)
    d2H_homo.append(d2H_th)


hessH_fcmnl5 = np.loadtxt(fcmnl_dir + "Fcmnl_b5_d2H_VV.txt")

# need to regroup the nonzero elements
n_prod_categories = ncat_men * ncat_women

d2H_fcmnl5 = []
# V was [iman, iwoman]
for iwoman in range(ncat_women):
    slice_woman = slice(iwoman, n_prod_categories, ncat_women)
    d2H_fcmnl5.append(hessH_fcmnl5[slice_woman, slice_woman])

# now each d2H[y] is a (ncat_men, ncat_men) matrix  of dmu(x|y)/dV(zy)


print_stars("Now ratio of Fcmnl5 to CS homo")
d2H_fratio = []
for iwoman in range(ncat_women):
    d2H_fratio.append(d2H_fcmnl5[iwoman] / d2H_homo[iwoman])

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
