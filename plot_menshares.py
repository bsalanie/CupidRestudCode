"""
Plot the shares of men in the BIC-preferred CS heteroskedastic model
"""

import numpy as np
from math import log

from cupid_utils import root_dir
from cupid_numpy_utils import npexp
from read_inputs import read_inputs

import matplotlib.pyplot as plt
plt.style.use('ggplot')


results_dir = root_dir + "Results/"
data_dir = root_dir + "Data/"
plots_dir = root_dir + "Plots/"

do_CS_homo = False
do_CS_homo_with_agd = False
do_CS_hetero = False
do_CS_hetero_with_agd = False
do_tau_search = False
do_CS_heteroxy = True
do_fcmnl = False


# first, read the data
mu_hat_norm, nx_norm, my_norm, sumw2, moments_hat_norm, market_norm, phibases = read_inputs(data_dir)
muxy_hat_norm, mux0_hat_norm, mu0y_hat_norm = mu_hat_norm.unpack()

# dimensions
ncat_men, ncat_women, n_bases = phibases.shape

# select the bases functions which sigma_x and tau_y depend on
indices_bases_sigma = [2]
indices_bases_tau = [0]
str_covariates = ""
for i in indices_bases_sigma:
    str_covariates += str(i)
str_covariates += "_"
for i in indices_bases_tau:
    str_covariates += str(i)
n_sigma = len(indices_bases_sigma)
n_tau = len(indices_bases_tau)
covariates_sigma = np.empty((ncat_men, n_sigma))
for i in range(n_sigma):
    covariates_sigma[:, i] = phibases[:, 0, indices_bases_sigma[i]]
covariates_tau = np.empty((ncat_women, n_tau))
for i in range(n_tau):
    covariates_tau[:, i] = phibases[0, :, indices_bases_tau[i]]
# the estimates
str_model = "CS_heteroXY_" + str_covariates
estimatesXY = np.loadtxt(results_dir + str_model + "_thetas.txt")

n_dist_params = n_sigma + n_tau
sigma_pars = estimatesXY[:n_sigma]
X_sigma = covariates_sigma @ sigma_pars
sigma_x = npexp(X_sigma)
tau_pars = estimatesXY[n_sigma:n_dist_params]
X_tau = covariates_tau @ tau_pars
tau_y = npexp(X_tau)


def men_shares_XY(age_man, age_woman):
    agem, agew = age_man-15, age_woman-15
    sigmax = sigma_x[agem]
    tauy = tau_y[agew]
    u_x = - sigmax * log(mux0_hat_norm[agem]/nx_norm[agem])
    v_y = - tauy * log(mu0y_hat_norm[agew]/my_norm[agew])
    return u_x/(u_x+v_y)


def men_shares_homo(age_man, age_woman):
    agem, agew = age_man-15, age_woman-15
    u_x = - log(mux0_hat_norm[agem]/nx_norm[agem])
    v_y = - log(mu0y_hat_norm[agew]/my_norm[agew])
    return u_x/(u_x+v_y)


shares_homo = np.zeros(25)
shares_hetero = np.zeros(25)
muxx = np.zeros(25)

for age in range(25):
    shares_homo[age] = men_shares_homo(age+15, age+15)
    shares_hetero[age] = men_shares_XY(age+15, age+15)
    muxx[age] = muxy_hat_norm[age, age]

muxx /= np.sum(muxx)

agebeg = 18
ageend = 25
ages = np.arange(agebeg, ageend + 1)

age_slice = slice(agebeg-16, ageend-15)

plt.clf()
fig, ax = plt.subplots()
ax.plot(ages, shares_homo[age_slice], '-ro', label='Homoskedastic')
ax.plot(ages, shares_hetero[age_slice], '-gx',
        label='Heteroskedastic')
ax.plot(ages, 4*muxx[age_slice], '--b', label='Ages at marriage')
ax.set_xlabel("Age of partners")
ax.set_ylabel("Man's share")
ax.set_ylim(0.0, 1.0)
ax.axhline(y=0.5, ls='--', c='k')
plt.legend(loc='upper center')
fig.tight_layout()
# plt.show()

plt.savefig(plots_dir + "MenShares" + str_covariates + ".eps")

