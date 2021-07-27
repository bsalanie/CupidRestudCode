import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

intermediate_data_dir = Path("./Intermediate")
output_data_dir = Path("./Output")

availables = pd.read_csv(intermediate_data_dir /
                         "ChooSiowAvailables.csv", dtype=int)
marriages = pd.read_csv(intermediate_data_dir /
                        "ChooSiowMarriages.csv", dtype=int)


# we need to de-age each partner by 1 or 2 years, depending on when the marriage is observed
year_observed = marriages['year'].values
aging_factor = year_observed - 10*np.floor(year_observed/10)
marriages['husband_age'] = marriages['husbagemarr'] - aging_factor
marriages['wife_age'] = marriages['wifeagemarr'] - aging_factor


def make_marriages(marriages: pd.DataFrame, age_start: int, age_end: int,
                   census_year: int, husband_reform: int, wife_reform: int) -> List:
    """ counts marriages by age cells

    with partners of ages between age_start and age_end at the date of the Census,
    and in states with a given reform status
    """
    n_ages = age_end-age_start+1
    ages_range = range(age_start, age_end + 1)
    # we only keep 1971 and 1972 marriages
    year_selection = marriages['year'].isin([census_year+1, census_year+2])
    # only if both partners are in a non-reform state
    state_selection = ((marriages['husbreform'] == husband_reform) & (
        marriages['wifereform'] == wife_reform))
    age_selection = (
        (marriages['husband_age'].isin(ages_range))
        & (marriages['wife_age'].isin(ages_range))
    )
    marriages_sel = marriages[state_selection & age_selection & year_selection]

    # go through age cells
    n_marriages = np.zeros((n_ages, n_ages))
    n_weighted_marriages = np.zeros((n_ages, n_ages))
    sumweights_sq = np.zeros((n_ages, n_ages))
    i_age = 0
    for h_age in ages_range:
        marr_h = marriages_sel[marriages_sel['husband_age'] == h_age]
        if marr_h.shape[0] > 0:
            j_age = 0
            for w_age in ages_range:
                marr_hw = marr_h[marr_h['wife_age'] == w_age]
                if marr_hw.shape[0] > 0:
                    weights = marr_hw['samplingweight'].values
                    n_weighted_marriages[i_age, j_age] = np.sum(weights)
                    n_marriages[i_age, j_age] = marr_hw.shape[0]
                    sumweights_sq[i_age, j_age] = np.sum(weights*weights)
                j_age += 1
        i_age += 1

    return n_weighted_marriages, n_marriages, sumweights_sq


n_weighted_marriages_70nN, n_marriages_70nN, sumw2_70nN = make_marriages(
    marriages, 16, 40, 1970, 0, 0)


np.savetxt(output_data_dir / "muxy70nN.txt", n_weighted_marriages_70nN)

# we need these next two to correct the number of availables
n_weighted_marriages_70nR, n_marriages_70nR, sumw2_70nR = make_marriages(
    marriages, 16, 40, 1970, 0, 1)
n_weighted_marriages_70rN, n_marriages_70rN, sumw2_70rN = make_marriages(
    marriages, 16, 40, 1970, 1, 0)


def make_availables70n(availables: pd.DataFrame, age_start: int, age_end: int) -> np.ndarray:
    """ create the series for available men and women in this age range.

    here we do it only for the (1970, non-reform) subset we use
    """
    n_ages = age_end-age_start+1
    ages_range = list(range(age_start, age_end + 1))
    # we only keep the 1970 Census
    year = availables['year']
    year_selection = (year == 1970)
    # only if the person is in a non-reform state
    state_selection = (availables['reform'] == 0)
    age_selection = (availables['age'].isin(ages_range))
    availables_sel = availables[state_selection &
                                age_selection & year_selection]

    total_availables = availables_sel.shape[0]
    total_weighted_availables = availables_sel['weight'].sum()

    n_available_men = np.zeros(n_ages)
    n_available_women = np.zeros(n_ages)
    n_weighted_available_men = np.zeros(n_ages)
    n_weighted_available_women = np.zeros(n_ages)
    sumweights_sq_men = np.zeros(n_ages)
    sumweights_sq_women = np.zeros(n_ages)

    i_age = 0
    for age in ages_range:
        available_age = availables_sel[availables_sel['age'] == age]
        available_h = available_age[available_age['sex'] == 1]
        weights_h = available_h['weight'].values
        # we need to subtract men who married a woman from a reform state
        n_available_men[i_age] = available_h.shape[0] - \
            np.sum(n_marriages_70nR[i_age, :])
        n_weighted_available_men[i_age] = np.sum(weights_h) - \
            np.sum(n_weighted_marriages_70nR[i_age, :])
        sumweights_sq_men[i_age] = np.sum(weights_h*weights_h) - \
            np.sum(sumw2_70nR[i_age, :])

        available_w = available_age[available_age['sex'] == 2]
        weights_w = available_w['weight'].values
        # we need to subtract women who married a man from a reform state
        n_available_women[i_age] = available_w.shape[0] - \
            np.sum(n_marriages_70rN[:, i_age])
        n_weighted_available_women[i_age] = np.sum(weights_w) - \
            np.sum(n_weighted_marriages_70rN[:, i_age])
        sumweights_sq_women[i_age] = np.sum(weights_w*weights_w) - \
            np.sum(sumw2_70rN[:, i_age])

        i_age += 1

    return n_available_men, n_weighted_available_men, sumweights_sq_men, \
        n_available_women, n_weighted_available_women, sumweights_sq_women


n_available_men, n_weighted_available_men, sumweights_sq_men, \
    n_available_women, n_weighted_available_women, sumweights_sq_women = \
    make_availables70n(availables, 16, 40)


np.savetxt(output_data_dir / "nx70n.txt", n_weighted_available_men)
np.savetxt(output_data_dir / "my70n.txt", n_weighted_available_women)


# we also need to compute the variance-covariance of the matching patterns
# again, we only do it for the 1970 Census in non-reform states

# total number of households in the sample
N_HOUSEHOLDS_OBS = np.sum(n_available_men) + \
    np.sum(n_available_women) - np.sum(n_marriages_70nN)


n_ages = n_available_men.size

# weighted matching patterns
muxy = n_weighted_marriages_70nN
mux0 = n_weighted_available_men - np.sum(muxy, 1)
mu0y = n_weighted_available_women - np.sum(muxy, 0)


# the variance-covariance matrix is 3 X 3 blocks
cov_muxy_muzt = np.zeros((n_ages, n_ages, n_ages, n_ages))
cov_muxy_muz0 = np.zeros((n_ages, n_ages, n_ages))
cov_muxy_mu0t = np.zeros((n_ages, n_ages, n_ages))
cov_mux0_muz0 = np.zeros((n_ages, n_ages))
cov_mux0_mu0y = np.zeros((n_ages, n_ages))
cov_mu0y_mu0t = np.zeros((n_ages, n_ages))

for x in range(n_ages):
    for y in range(n_ages):
        cov_muxy_muzt[x, y, :, :] = -muxy[x, y]*muxy/N_HOUSEHOLDS_OBS
        cov_muxy_muzt[x, y, x, y] += sumw2_70nN[x, y]
        cov_muxy_muz0[x, y, :] = -muxy[x, y]*mux0/N_HOUSEHOLDS_OBS
        cov_muxy_mu0t[x, y, :] = -muxy[x, y]*mu0y/N_HOUSEHOLDS_OBS

for x in range(n_ages):
    cov_mux0_muz0[x, :] = -mux0[x]*mux0/N_HOUSEHOLDS_OBS
    cov_mux0_muz0[x, x] += sumweights_sq_men[x]
    cov_mux0_mu0y[x, :] = -mux0[x]*mu0y/N_HOUSEHOLDS_OBS

for y in range(n_ages):
    cov_mu0y_mu0t[y, :] = -mu0y[y]*mu0y/N_HOUSEHOLDS_OBS
    cov_mu0y_mu0t[y, y] += sumweights_sq_women[y]

# we store all covariances in a large matrix
n_ages_sq = n_ages*n_ages
n_var = n_ages_sq+2*n_ages
var_mus = np.zeros((n_var, n_var))

slice_xy = slice(0, n_ages_sq)
slice_x0 = slice(n_ages_sq, n_ages_sq + n_ages)
slice_0y = slice(n_ages_sq + n_ages, n_var)
# we start with the covariances of muxy
ixy = 0
for x in range(n_ages):
    for y in range(n_ages):
        var_mus[ixy, slice_xy] = cov_muxy_muzt[x, y, :, :].reshape(n_ages_sq)
        var_mus[ixy, slice_x0] = \
            cov_muxy_muz0[x, y, :]
        var_mus[ixy, slice_0y] = \
            cov_muxy_mu0t[x, y, :]
        ixy += 1

# then the covariances of mux0
var_mus[slice_x0, slice_xy] = \
    var_mus[slice_xy, slice_x0].T
for x in range(n_ages):
    var_mus[n_ages_sq+x, slice_x0] = cov_mux0_muz0[x, :]
    var_mus[n_ages_sq+x, slice_0y] = cov_mux0_mu0y[x, :]

# finally, the covariances of mu0y
var_mus[slice_0y, slice_xy] = \
    var_mus[slice_xy, slice_0y].T
var_mus[slice_0y, slice_x0] = \
    var_mus[slice_x0, slice_0y].T
for y in range(n_ages):
    var_mus[n_ages_sq+n_ages+y, slice_0y] = \
        cov_mu0y_mu0t[y, :]

np.savetxt(output_data_dir / "varmus70nN.txt", var_mus)


# finally, we create the basis functions for the surplus matrix Phi

# we demean and standardize the x and y  variables
x = np.arange(n_ages)
y = np.arange(n_ages)
X = (x-np.mean(x))/np.std(x)
Y = (y-np.mean(y))/np.std(y)


def make_basis_function(m: int, n: int, D: bool = False) \
        -> Tuple[np.ndarray, str]:
    """ create the basis function (X**m) * (Y**n) 
        multiplied by 1(x>=y) id D is True
    """
    phi_base = np.outer(np.power(X, m), np.power(Y, n))
    name_base = f"X^{m} Y^{n}"
    if D:
        x_larger = np.ones((n_ages, n_ages))
        for x in range(n_ages):
            x_larger[x, :x] = 0.0
        phi_base *= x_larger
        name_base += " D"
    return phi_base, name_base


def make_phibases(M: int, N: int) \
        -> Tuple[np.ndarray, int, List[str]]:
    """ create all terms (X**m) * (Y**n) * (D**p)
    for 0<=m<=M, 0<=n<=N, and p=0, 1
    """
    n_bases = 2*(M+1)*(N+1)
    phi_bases = np.zeros((n_ages, n_ages, n_bases))
    names_bases = [""]*n_bases
    i = 0
    for m in range(M+1):
        for n in range(N+1):
            phi_bases[:, :, i], names_bases[i] = make_basis_function(m, n)
            phi_bases[:, :, i+1], names_bases[i+1] = \
                make_basis_function(m, n, D=True)
            i += 2
    return phi_bases, n_bases, names_bases


# the specification we selected
phibases24, n_bases, names_bases = make_phibases(2, 4)
# to store it, we make it a matrix
phimatrix24 = np.zeros((n_ages*n_ages, n_bases))
for i in range(n_bases):
    phimatrix24[:, i] = phibases24[:, :, i].reshape(n_ages*n_ages)
np.savetxt(output_data_dir / "phibases24.txt", phimatrix24)
with open(output_data_dir / "names_bases24.txt", "w") as f:
    for i in range(n_bases):
        f.write(names_bases[i] + "\n")
