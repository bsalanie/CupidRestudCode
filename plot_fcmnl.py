"""
Plot the simulated matching patterns for FCMNL
"""

import numpy as np
from math import log

from cupid_classes import CupidParamsFcmnl
from cupid_utils import root_dir, N_HOUSEHOLDS_OBS, print_stars
from read_inputs import read_inputs
from solve_for_mus import mus_fcmnl_and_maybe_grad_agd
from fcmnl import derivs_GplusH_fcmnl, make_b5

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
homo_dir = results_dir + "Homoskedastic/"
fcmnl_dir = results_dir + "Fcmnl/"

str_title_homo = "Choo-Siow homoskedastic"
str_title_fcmnl = "Fcmnl"

# read the simulated mus
muxy_norm_homo = np.loadtxt(homo_dir + "CS_homoskedastic_muxy_norm.txt")
mux0_norm_homo = np.loadtxt(homo_dir + "CS_homoskedastic_mux0_norm.txt")
mu0y_norm_homo = np.loadtxt(homo_dir + "CS_homoskedastic_mu0y_norm.txt")
muxy_norm_fcmnl5 = np.loadtxt(fcmnl_dir + "Fcmnl_b5_muxy_norm.txt")
mux0_norm_fcmnl5 = np.loadtxt(fcmnl_dir + "Fcmnl_b5_mux0_norm.txt")
mu0y_norm_fcmnl5 = np.loadtxt(fcmnl_dir + "Fcmnl_b5_mu0y_norm.txt")

# compute the margins
nx_homo = np.sum(muxy_norm_homo, 1) + mux0_norm_homo
my_homo = np.sum(muxy_norm_homo, 0) + mu0y_norm_homo
nx_fcmnl5 = np.sum(muxy_norm_fcmnl5, 1) + mux0_norm_fcmnl5
my_fcmnl5 = np.sum(muxy_norm_fcmnl5, 0) + mu0y_norm_fcmnl5
# and the conditional matching patterns
mugivenx_homo = muxy_norm_homo / nx_homo.reshape((-1, 1))
mugiveny_homo = muxy_norm_homo / my_homo
mugivenx_fcmnl5 = muxy_norm_fcmnl5 / nx_fcmnl5.reshape((-1, 1))
mugiveny_fcmnl5 = muxy_norm_fcmnl5 / my_fcmnl5

# compute d2H/(dmu dmu') for the Choo and Siow homoskedastic model
#   we use the closed form formula
d2H_homo = []
for iwoman in range(ncat_women):
    muxy_woman = muxy_norm_homo[:, iwoman]
    mugiveny_woman = mugiveny_homo[:, iwoman]
    d2H_th = np.diag(muxy_woman) - np.outer(muxy_woman, mugiveny_woman)
    d2H_homo.append(d2H_th)

# for the FCMNL model we need more work
#  we load the U at the stable matching
U_fcmnl5 = np.loadtxt(fcmnl_dir + "Fcmnl_b5_U.txt")
thetas_fcmnl5 = np.loadtxt(fcmnl_dir + "Fcmnl_b5_thetas.txt")
Phi_fcmnl5 = phibases @ thetas_fcmnl5[2:]
fcmnl_params_norm = CupidParamsFcmnl(men_margins=nx_norm, women_margins=my_norm,
                                     observed_matching=mu_hat_norm,
                                     bases_surplus=phibases,
                                     mus_and_maybe_grad=mus_fcmnl_and_maybe_grad_agd,
                                     make_b=make_b5,
                                     n_pars_b_men=0,
                                     n_pars_b_women=2)
pars_b_men = np.array([])
pars_b_women = thetas_fcmnl5[:2]
derivs_fcmnl5 = derivs_GplusH_fcmnl(U_fcmnl5, fcmnl_params_norm, Phi_fcmnl5,
                        pars_b_men, pars_b_women,
                        derivs=2)
hessH_fcmnl5 = derivs_fcmnl5.hessians[3]

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
iwoman = 0
for i in range(nrows):
    for j in range(ncols):
        fratio = d2H_homo[iwoman] / muxy_norm_homo[:, iwoman].reshape((-1, 1))
        plot_matching(ax[i, j], fratio, iwoman, ncat_women)
        iwoman += 1

plt.savefig(plots_dir + "Semi_elasticities_women_CShomo.eps")

plt.clf()

fig, ax = plt.subplots(nrows, ncols, constrained_layout=True)
iwoman = 0
for i in range(nrows):
    for j in range(ncols):
        fratio = d2H_fcmnl5[iwoman] / muxy_norm_fcmnl5[:, iwoman].reshape((-1, 1))
        plot_matching(ax[i, j], fratio, iwoman, ncat_women)
        iwoman += 1

plt.savefig(plots_dir + "Semi_elasticities_women_Fcmnl.eps")
