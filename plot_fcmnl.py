"""
Plot the simulated matching patterns for FCMNL
"""

import numpy as np

from cupid_classes import CupidParamsFcmnl
from cupid_utils import root_dir
from read_inputs import read_inputs
from solve_for_mus import mus_fcmnl_and_maybe_grad_agd
from fcmnl import derivs_GplusH_fcmnl, make_b5

import matplotlib.pyplot as plt

plt.style.use('ggplot')

results_dir = root_dir + "Results/"
data_dir = root_dir + "Data/"
plots_dir = root_dir + "Plots/"

# first, read the data
mu_hat_norm, nx_norm, my_norm, phibases, varmus = read_inputs(data_dir)
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


# the semi-elasticities
def semi_elasticities(d2H, muxy_norm, iwoman):
    semi_elast = d2H[iwoman] / muxy_norm[:, iwoman].reshape((-1, 1))
    np.fill_diagonal(semi_elast, 0.0)
    return semi_elast

# ages surrounding a woman's age
def get_slice(iwoman):
    age_min = 16 + max(0, iwoman - 5)
    age_max = 16 + min(ncat_women, iwoman + 5)
    return slice(age_min - 16, age_max - 16 + 1)


# to find minimum of minima and maximum of maxima
#  so that the subplots and the color bar use the same colors
mini_min = 0.0
maxi_max = 0.0
iwoman = 0
for irow in range(6):
    semi_elast_homo = semi_elasticities(d2H_homo, muxy_norm_homo, iwoman)
    semi_elast_fcmnl5 = semi_elasticities(d2H_fcmnl5, muxy_norm_fcmnl5, iwoman)
    slice_woman = get_slice(iwoman)
    mini_homo = np.min(semi_elast_homo[slice_woman, slice_woman])
    mini_fcmnl5 = np.min(semi_elast_fcmnl5[slice_woman, slice_woman])
    mini_min = min(mini_min, mini_homo, mini_fcmnl5)
    maxi_homo = np.max(semi_elast_homo[slice_woman, slice_woman])
    maxi_fcmnl5 = np.max(semi_elast_fcmnl5[slice_woman, slice_woman])
    maxi_max = max(maxi_max, maxi_homo, maxi_fcmnl5)


def plot_semi_elasticities(semi_elast, axi):
    return axi.imshow(semi_elast, origin='lower',
                      vmin=mini_min, vmax=maxi_max,
                      extent=(-5, 5, -5, 5),
                      aspect='auto', cmap='viridis')


fig, ax = plt.subplots(6, 3, figsize=(6, 6))
iwoman = 0
for irow in range(6):
    semi_elast_homo = semi_elasticities(d2H_homo, muxy_norm_homo, iwoman)
    semi_elast_fcmnl5 = semi_elasticities(d2H_fcmnl5, muxy_norm_fcmnl5, iwoman)
    slice_woman = get_slice(iwoman)
    # left column
    im = plot_semi_elasticities(semi_elast_homo[slice_woman, slice_woman], ax[irow, 0])
    # right column
    im = plot_semi_elasticities(semi_elast_fcmnl5[slice_woman, slice_woman], ax[irow, 2])

    # we create the middle column
    ax[irow, 1].axis("off")
    ax[irow, 1].grid("off")
    ax[irow, 1].text(0.25, 0.45, "Age " + str(iwoman + 16))

    if irow != 5:
        for i in [0, 2]:
            ax[irow, i].set_xticks([])
            ax[irow, i].set_yticks([])

    iwoman += 1

for i in [0, 2]:
    ax[5, i].set_xticks([-5, 0, 5])
    ax[5, i].set_xticklabels(
        [str(i) for i in [16, 21, 26]])
ax[5, 0].set_yticks([-5, 0, 5])
ax[5, 0].set_yticklabels(
    [str(i) for i in [16, 21, 26]])
ax[5, 2].set_yticks([])

ax[0,0].set_title("Choo-Siow homoskedastic", fontsize=12)
ax[0,2].set_title("FC-MNL", fontsize=12)

# add space for colour bar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.83, 0.15, 0.04, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.savefig(plots_dir + "semi_elasticities.eps")
