"""
G and H functions for FC-MNL
"""

from math import log
import numpy as np
from typing import Tuple, Optional, Union

import multiprocessing as mp

from cupid_classes import GHReturn, DerivsGHReturn, CupidParamsFcmnl

from cupid_utils import EULER_CONSTANT, print_stars, describe_array, bs_error_abort, GRADIENT_STEP
from cupid_numpy_utils import npexp, der_npexp, der2_npexp, nppow, der_nppow, der2_nppow


def fc_dist(ncat_partner: int) -> np.ndarray:
    """
    make a 2-dim array of distances |age1 - age2| betwen partners for every type

    :param int ncat_partner: number of types of partners

    :return: array of shape (ncat_partner, ncat_partner)
    """
    ages_partner = np.arange(ncat_partner, dtype=float)
    distances = np.abs(np.subtract.outer(ages_partner, ages_partner))
    np.fill_diagonal(distances, 1.0)  # or anything but zero
    return distances


#
#
#  each _b_XX function fills the b matrix and its derivatives: with 0, 1, 2 parameters
#
#
def _b_zero(ncat: int, ncat_partner: int) -> np.ndarray:
    b = np.zeros((ncat, ncat_partner, ncat_partner))
    for i in range(ncat):
        np.fill_diagonal(b[i, :, :], 1.0)
    return b


def _b_one(pars_b: np.ndarray, distances: np.ndarray, ncat: int, ncat_partner: int) -> Tuple[np.ndarray, np.ndarray]:
    par_b0 = pars_b[0]
    b = np.zeros((ncat, ncat_partner, ncat_partner))
    db = np.zeros((ncat, ncat_partner, ncat_partner, 1))

    for i in range(ncat):
        b[i, :, :] = par_b0 / distances
        db[i, :, :, 0] = 1.0 / distances
        # we need 1.0 on the diagonals, with 0 derivatives
        np.fill_diagonal(b[i, :, :], 1.0)
        np.fill_diagonal(db[i, :, :, 0], 0.0)
    return b, db


def _b_two(pars_b: np.ndarray, distances: np.ndarray, ncat: int, ncat_partner: int) -> Tuple[np.ndarray, np.ndarray]:
    par_b0 = pars_b[0]
    par_b1 = pars_b[1]
    b = np.zeros((ncat, ncat_partner, ncat_partner))
    db = np.zeros((ncat, ncat_partner, ncat_partner, 2))
    xvec = np.arange(ncat) / (ncat - 1.0)
    par_bx = xvec * par_b1 + (1.0 - xvec) * par_b0

    for i in range(ncat):
        b[i, :, :] = par_bx[i] / distances
        xi = xvec[i]
        db[i, :, :, 0] = (1.0 - xi) / distances
        db[i, :, :, 1] = xi / distances
        # we need 1.0 on the diagonals, with 0 derivatives
        np.fill_diagonal(b[i, :, :], 1.0)
        np.fill_diagonal(db[i, :, :, 0], 0.0)
        np.fill_diagonal(db[i, :, :, 1], 0.0)
    return b, db

#
#
# each make_bX function compute b = p/fc_dist and its derivatives for both sides, for specification bX
#
#

def make_b0(pars_b_men: np.ndarray, pars_b_women: np.ndarray, ncat_men: int, ncat_women: int) \
        -> Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray, Union[np.ndarray, None]]:
    """
    compute b = p/fc_dist and its derivatives for both sides, orders (0,0)

    :param np.ndarray pars_b_men: parameters of b for men

    :param np.ndarray pars_b_women: parameters of b for women

    :param int ncat_men: number of categories of men

    :param int ncat_women: number of categories of women

    :return: values and first derivatives of b for each (x,y,t) or (y,x,z)
    """
    n_pars_b_men, n_pars_b_women = pars_b_men.size, pars_b_women.size

    if n_pars_b_men != 0:
        bs_error_abort(f"we need n_pars_b_men = 0 not {n_pars_b_men}")
    if n_pars_b_women != 0:
        bs_error_abort(f"we need n_pars_b_women = 0 not {n_pars_b_women}")

    b_men = _b_zero(ncat_men, ncat_women)
    b_women = _b_zero(ncat_women, ncat_men)

    return b_men, None, b_women, None


def make_b1(pars_b_men: np.ndarray, pars_b_women: np.ndarray, ncat_men: int, ncat_women: int) \
        -> Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray, Union[np.ndarray, None]]:
    """
    compute b = p/fc_dist and its derivatives for both sides, orders (1,0)

    :param np.ndarray pars_b_men: parameters of b for men

    :param np.ndarray pars_b_women: parameters of b for women

    :param int ncat_men: number of categories of men

    :param int ncat_women: number of categories of women

    :return: values and first derivatives of b for each (x,y,t) or (y,x,z)
    """
    n_pars_b_men, n_pars_b_women = pars_b_men.size, pars_b_women.size

    if n_pars_b_men != 1:
        bs_error_abort(f"we need n_pars_b_men = 1 not {n_pars_b_men}")
    if n_pars_b_women != 0:
        bs_error_abort(f"we need n_pars_b_women = 0 not {n_pars_b_women}")

    fc_dist_women = fc_dist(ncat_women)

    b_men, db_men = _b_one(pars_b_men, fc_dist_women, ncat_men, ncat_women)
    b_women = _b_zero(ncat_women, ncat_men)

    return b_men, db_men, b_women, None


def make_b2(pars_b_men: np.ndarray, pars_b_women: np.ndarray, ncat_men: int, ncat_women: int) \
        -> Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray, Union[np.ndarray, None]]:
    """
    compute b = p/fc_dist and its derivatives for both sides, orders (0,1)

    :param np.ndarray pars_b_men: parameters of b for men

    :param np.ndarray pars_b_women: parameters of b for women

    :param int ncat_men: number of categories of men

    :param int ncat_women: number of categories of women

    :return: values and first derivatives of b for each (x,y,t) or (y,x,z)
    """
    n_pars_b_men, n_pars_b_women = pars_b_men.size, pars_b_women.size

    if n_pars_b_men != 0:
        bs_error_abort(f"we need n_pars_b_men = 0 not {n_pars_b_men}")
    if n_pars_b_women != 1:
        bs_error_abort(f"we need n_pars_b_women = 1 not {n_pars_b_women}")

    fc_dist_men = fc_dist(ncat_men)

    b_women, db_women = _b_one(pars_b_women, fc_dist_men, ncat_women, ncat_men)
    b_men = _b_zero(ncat_men, ncat_women)

    return b_men, None, b_women, db_women


def make_b3(pars_b_men: np.ndarray, pars_b_women: np.ndarray, ncat_men: int, ncat_women: int) \
        -> Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray, Union[np.ndarray, None]]:
    """
    compute b = p/fc_dist and its derivatives for both sides, orders (2,0)

    :param np.ndarray pars_b_men: parameters of b for men

    :param np.ndarray pars_b_women: parameters of b for women

    :param int ncat_men: number of categories of men

    :param int ncat_women: number of categories of women

    :return: values and first derivatives of b for each (x,y,t) or (y,x,z)
    """
    n_pars_b_men, n_pars_b_women = pars_b_men.size, pars_b_women.size

    if n_pars_b_men != 2:
        bs_error_abort(f"we need n_pars_b_men = 2 not {n_pars_b_men}")
    if n_pars_b_women != 0:
        bs_error_abort(f"we need n_pars_b_women = 0 not {n_pars_b_women}")

    fc_dist_women = fc_dist(ncat_women)

    b_men, db_men = _b_two(pars_b_men, fc_dist_women, ncat_men, ncat_women)
    b_women = _b_zero(ncat_women, ncat_men)

    return b_men, db_men, b_women, None


def make_b4(pars_b_men: np.ndarray, pars_b_women: np.ndarray, ncat_men: int, ncat_women: int) \
        -> Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray, Union[np.ndarray, None]]:
    """
    compute b = p/fc_dist and its derivatives for both sides, orders (1,1)

    :param np.ndarray pars_b_men: parameters of b for men

    :param np.ndarray pars_b_women: parameters of b for women

    :param int ncat_men: number of categories of men

    :param int ncat_women: number of categories of women

    :return: values and first derivatives of b for each (x,y,t) or (y,x,z)
    """
    n_pars_b_men, n_pars_b_women = pars_b_men.size, pars_b_women.size

    if n_pars_b_men != 1:
        bs_error_abort(f"we need n_pars_b_men = 1 not {n_pars_b_men}")
    if n_pars_b_women != 1:
        bs_error_abort(f"we need n_pars_b_women = 1 not {n_pars_b_women}")

    fc_dist_women = fc_dist(ncat_women)
    fc_dist_men = fc_dist(ncat_men)

    b_men, db_men = _b_one(pars_b_men, fc_dist_women, ncat_men, ncat_women)
    b_women, db_women = _b_one(pars_b_women, fc_dist_men, ncat_women, ncat_men)

    return b_men, db_men, b_women, db_women


def make_b5(pars_b_men: np.ndarray, pars_b_women: np.ndarray, ncat_men: int, ncat_women: int) \
        -> Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray, Union[np.ndarray, None]]:
    """
    compute b = p/fc_dist and its derivatives for both sides, orders (0,2)

    :param np.ndarray pars_b_men: parameters of b for men

    :param np.ndarray pars_b_women: parameters of b for women

    :param int ncat_men: number of categories of men

    :param int ncat_women: number of categories of women

    :return: values and first derivatives of b for each (x,y,t) or (y,x,z)
    """
    n_pars_b_men, n_pars_b_women = pars_b_men.size, pars_b_women.size

    if n_pars_b_men != 0:
        bs_error_abort(f"we need n_pars_b_men = 0 not {n_pars_b_men}")
    if n_pars_b_women != 2:
        bs_error_abort(f"we need n_pars_b_women = 2 not {n_pars_b_women}")

    fc_dist_men = fc_dist(ncat_men)

    b_men = _b_zero(ncat_men, ncat_women)
    b_women, db_women = _b_two(pars_b_women, fc_dist_men, ncat_women, ncat_men)

    return b_men, None, b_women, db_women


def make_b6(pars_b_men: np.ndarray, pars_b_women: np.ndarray, ncat_men: int, ncat_women: int) \
        -> Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray, Union[np.ndarray, None]]:
    """
    compute b = p/fc_dist and its derivatives for both sides, orders (2,1)

    :param np.ndarray pars_b_men: parameters of b for men

    :param np.ndarray pars_b_women: parameters of b for women

    :param int ncat_men: number of categories of men

    :param int ncat_women: number of categories of women

    :return: values and first derivatives of b for each (x,y,t) or (y,x,z)
    """
    n_pars_b_men, n_pars_b_women = pars_b_men.size, pars_b_women.size

    if n_pars_b_men != 2:
        bs_error_abort(f"we need n_pars_b_men = 2 not {n_pars_b_men}")
    if n_pars_b_women != 1:
        bs_error_abort(f"we need n_pars_b_women = 1 not {n_pars_b_women}")

    fc_dist_men = fc_dist(ncat_men)
    fc_dist_women = fc_dist(ncat_women)

    b_men, db_men = _b_two(pars_b_men, fc_dist_women, ncat_men, ncat_women)
    b_women, db_women = _b_one(pars_b_women, fc_dist_men, ncat_women, ncat_men)

    return b_men, db_men, b_women, db_women


def make_b7(pars_b_men: np.ndarray, pars_b_women: np.ndarray, ncat_men: int, ncat_women: int) \
        -> Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray, Union[np.ndarray, None]]:
    """
    compute b = p/fc_dist and its derivatives for both sides, orders (1,2)

    :param np.ndarray pars_b_men: parameters of b for men

    :param np.ndarray pars_b_women: parameters of b for women

    :param int ncat_men: number of categories of men

    :param int ncat_women: number of categories of women

    :return: values and first derivatives of b for each (x,y,t) or (y,x,z)
    """
    n_pars_b_men, n_pars_b_women = pars_b_men.size, pars_b_women.size

    if n_pars_b_men != 1:
        bs_error_abort(f"we need n_pars_b_men = 1 not {n_pars_b_men}")
    if n_pars_b_women != 2:
        bs_error_abort(f"we need n_pars_b_women = 2 not {n_pars_b_women}")

    fc_dist_men = fc_dist(ncat_men)
    fc_dist_women = fc_dist(ncat_women)

    b_men, db_men = _b_one(pars_b_men, fc_dist_women, ncat_men, ncat_women)
    b_women, db_women = _b_two(pars_b_women, fc_dist_men, ncat_women, ncat_men)

    return b_men, db_men, b_women, db_women


def make_b8(pars_b_men: np.ndarray, pars_b_women: np.ndarray, ncat_men: int, ncat_women: int) \
        -> Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray, Union[np.ndarray, None]]:
    """
    compute b = p/fc_dist and its derivatives for both sides, orders (2,2)

    :param np.ndarray pars_b_men: parameters of b for men

    :param np.ndarray pars_b_women: parameters of b for women

    :param int ncat_men: number of categories of men

    :param int ncat_women: number of categories of women

    :return: values and first derivatives of b for each (x,y,t) or (y,x,z)
    """
    n_pars_b_men, n_pars_b_women = pars_b_men.size, pars_b_women.size

    if n_pars_b_men != 2:
        bs_error_abort(f"we need n_pars_b_men = 2 not {n_pars_b_men}")
    if n_pars_b_women != 2:
        bs_error_abort(f"we need n_pars_b_women = 2 not {n_pars_b_women}")

    fc_dist_men = fc_dist(ncat_men)
    fc_dist_women = fc_dist(ncat_women)

    b_men, db_men = _b_two(pars_b_men, fc_dist_women, ncat_men, ncat_women)
    b_women, db_women = _b_two(pars_b_women, fc_dist_men, ncat_women, ncat_men)

    return b_men, db_men, b_women, db_women


def fcmnl_generator(U: np.ndarray, b: np.ndarray, sigma: float, tau: float,
                    derivs: Optional[int] = None) -> GHReturn:
    """
    returns the generating function of FC-MNL

    :param np.ndarray U: (ncat_partner) array of arguments

    :param np.ndarray b: matrix (ncat_partner, ncat_partner) symmetric and positive with diagonal = 1

    :param float sigma: the FC-MNL parameter :math:`\\sigma`

    :param float tau: the FC-MNL parameter :math:`\\tau`

    :param int derivs: how many derivatives wrt U and b we need

    :return: a GH_return instance (the value of the generating function, with derivatives wrt U and b as requested)
    """
    assert U.ndim == 1
    assert b.ndim == 2
    ncat_partner = U.size
    assert b.shape[0] == b.shape[1] == ncat_partner

    sig1 = 1.0 / sigma
    sig1_vec = np.full(ncat_partner, sig1)
    U1 = U * sig1_vec
    tausig = tau * sigma
    expU1 = npexp(U1)
    ave = np.add.outer(expU1, expU1) / 2.0
    ave_pow = nppow(ave, tausig)
    val_generator = np.sum(b * ave_pow) + 1.0
    resus = GHReturn(value=val_generator, derivs=derivs)
    if derivs is not None:
        order_max = derivs
        if order_max >= 1:
            dave_pow, _ = der_nppow(ave, tausig)
            dexpU1 = der_npexp(U1)
            der_generator_U = sig1 * dexpU1 * \
                (np.sum(b * dave_pow, 0) + np.sum(b * dave_pow, 1)) / 2.0
            der_generator_b = ave_pow
            resus.gradients = [der_generator_b, der_generator_U, None, None]
        if order_max >= 2:
            d2ave_pow, _, _ = der2_nppow(ave, tausig)
            der2_generator_bU = np.zeros(
                (ncat_partner, ncat_partner, ncat_partner))
            for j in range(ncat_partner):
                ddj = dave_pow[j, :] * dexpU1[j]
                der2_generator_bU[j, :, j] = ddj
            for k in range(ncat_partner):
                ddk = dave_pow[:, k] * dexpU1[k]
                der2_generator_bU[:, k, k] = ddk
            for j in range(ncat_partner):
                der2_generator_bU[j, j, j] = 2.0 * dave_pow[j, j] * dexpU1[j]
            der2_generator_bU *= (sig1 / 2.0)
            der2_generator_UU = d2ave_pow * b * \
                np.outer(dexpU1, dexpU1) * (sig1 * sig1 / 2.0)
            d2expU1 = der2_npexp(U1)
            for j in range(ncat_partner):
                bj = b[j, :]
                dexpU1j = dexpU1[j]
                d2expU1j = d2expU1[j]
                davej = dave_pow[j, :]
                d2avej = d2ave_pow[j, :]
                diagj = np.sum(
                    bj * (d2avej * dexpU1j * dexpU1j + 2.0 * davej * d2expU1j))
                der2_generator_UU[j, j] += (diagj / (2.0 * sigma * sigma))
            resus.hessians = [der2_generator_bU, der2_generator_UU]
        if order_max > 2:
            bs_error_abort("no derivatives of order > 2!")
    return resus


def value_Gx(Uman: np.ndarray, b_man: np.ndarray, sigma: float, tau: float) -> float:
    """
    computes the value of :math:`G_x(U_{x\\cdot})` for one type of man

    :param np.ndarray Uman: the values of :math:`U_{x\\cdot}`

    :param np.ndarray b_man: the values of b for this type

    :param float sigma: the FC-MNL sigma

    :param float tau: the FC-MNL tau

    :return: the value of :math:`G_x(U_{x\\cdot})`
    """
    resus_x = fcmnl_generator(Uman, b_man, sigma, tau, derivs=0)
    Gx_val = log(resus_x.value) / tau + EULER_CONSTANT
    return Gx_val


def value_Hy(Vwoman: np.ndarray, b_woman: np.ndarray, sigma: float, tau: float) -> float:
    """
    computes the value of :math:`H_y(V_{\\cdot y})` for one type of woman

    :param np.ndarray Vwoman: the values of :math:`V_{\\cdot y}`

    :param np.ndarray b_woman: the values of b for this type

    :param float sigma: the FC-MNL sigma

    :param float tau: the FC-MNL tau

    :return: the value of :math:`H_y(V_{\\cdot y})`
    """
    resus_y = fcmnl_generator(Vwoman, b_woman, sigma, tau, derivs=0)
    Hy_val = log(resus_y.value) / tau + EULER_CONSTANT
    return Hy_val


def gradients_Gx(Uman: np.ndarray, b_man: np.ndarray, db_man: Union[np.ndarray, None],
                 nb_man: np.ndarray, sigma: float, tau: float) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    computes the gradients of :math:`G_x(U_{x \\cdot})` for one type of man

    :param np.ndarray Uman: the values of :math:`U_{x \\cdot}`

    :param np.ndarray b_man: the values of b for this type

    :param np.ndarray db_man: the values of db for this type (if n_pars_b_men > 0)

    :param float nb_man: the number of men of this type

    :param float sigma: the FC-MNL sigma

    :param float tau: the FC-MNL tau

    :return: a 3-uple with the value of :math:`G_x(U_{x \\cdot})` and its derivatives wrt the parameters of b, \
        and wrt :math:`U_{x \\cdot}`
    """
    resus_x = \
        fcmnl_generator(Uman, b_man, sigma, tau, derivs=1)
    value_x = resus_x.value
    Gx_val = log(value_x) / tau + EULER_CONSTANT
    Gval = nb_man * Gx_val
    fac_x = 1.0 / tau / value_x
    nx_fx = nb_man * fac_x
    dgenfun_dU = resus_x.gradients[1]
    gradG_U = nx_fx * dgenfun_dU
    if db_man is not None:
        dgenfun_db = resus_x.gradients[0]
        dgenfun_db_p = np.einsum('ijk, ij->k', db_man, dgenfun_db)
        gradG_d = nx_fx * dgenfun_db_p
    else:
        gradG_d = None
    return Gval, gradG_d, gradG_U


def gradients_Hy(Vwoman: np.ndarray, b_woman: np.ndarray, db_woman: Union[np.ndarray, None],
                 nb_woman: np.ndarray, sigma: float, tau: float) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    computes the gradients of :math:`H_y(V_{\\cdot y})` for one type of woman

    :param np.ndarray Vwoman: the values of :math:`V_{\\cdot y}`

    :param np.ndarray b_woman: the values of b for this type

    :param np.ndarray db_woman: the values of db for this type (if n_pars_b_women > 0)

    :param float nb_woman: the number of women of this type

    :param float sigma: the FC-MNL sigma

    :param float tau: the FC-MNL tau

    :return: a 3-uple with the value of :math:`H_y(V_{\\cdot y})` and its derivatives wrt the parameters of b, \
    and wrt :math:`V_{\\cdot y}`
    """
    resus_y = \
        fcmnl_generator(Vwoman, b_woman, sigma, tau, derivs=1)
    value_y = resus_y.value
    Hy_val = log(value_y) / tau + EULER_CONSTANT
    Hval = nb_woman * Hy_val
    fac_y = 1.0 / tau / value_y
    my_fy = nb_woman * fac_y
    dgenfun_dV = resus_y.gradients[1]
    gradH_V = my_fy * dgenfun_dV
    if db_woman is not None:
        dgenfun_db = resus_y.gradients[0]
        dgenfun_db_p = np.einsum('ijk, ij->k', db_woman, dgenfun_db)
        gradH_d = my_fy * dgenfun_db_p
    else:
        gradH_d = None

    return Hval, gradH_d, gradH_V


def hessians_Gx(Uman: np.ndarray, b_man: np.ndarray, db_man: Union[np.ndarray, None],
                nb_man: np.ndarray, sigma: float, tau: float) \
        -> Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray, Union[np.ndarray, None], np.ndarray]:
    """
    computes the hessians of :math:`G_x(U_{x \\cdot})` for one type of man

    :param np.ndarray Uman: the values of :math:`U_{x \\cdot}`

    :param np.ndarray b_man: the values of b for this type

    :param np.ndarray db_man: the values of db for this type (if n_pars_b_men > 0)

    :param float nb_man: the number of men of this type

    :param float sigma: the FC-MNL sigma

    :param float tau: the FC-MNL tau

    :return: a 5-uple with the value of :math:`G_x(U_{x \\cdot})` and its derivatives wrt the parameters of b, \
        and wrt :math:`U_{x \\cdot}`, and the hessians in (d,U) and (U,U)
    """
    resus_x = \
        fcmnl_generator(Uman, b_man, sigma, tau, derivs=2)
    value_x = resus_x.value
    Gx_val = log(value_x) / tau + EULER_CONSTANT
    Gval = nb_man * Gx_val

    fac_x = 1.0 / tau / value_x
    nx_fx = nb_man * fac_x
    fac_x2 = fac_x / value_x
    nx_fx2 = nb_man * fac_x2

    dgenfun_dU = resus_x.gradients[1]
    gradG_U = nx_fx * dgenfun_dU
    d2genfun_dUU = resus_x.hessians[1]
    d2G_UU = nx_fx * d2genfun_dUU - nx_fx2 * np.outer(dgenfun_dU, dgenfun_dU)

    if db_man is not None:
        dgenfun_db = resus_x.gradients[0]
        dgenfun_db_p = np.einsum('ijk, ij->k', db_man, dgenfun_db)
        gradG_d = nx_fx * dgenfun_db_p
        d2genfun_dbU = resus_x.hessians[0]
        d2genfun_dbU_p = np.einsum('ijk, ijl->kl', db_man, d2genfun_dbU)
        d2G_dU = nx_fx * d2genfun_dbU_p - nx_fx2 * \
            np.outer(dgenfun_db_p, dgenfun_dU)
        return (Gval, gradG_d, gradG_U, d2G_dU, d2G_UU)
    else:
        return (Gval, None, gradG_U, None, d2G_UU)


def hessians_Hy(Vwoman: np.ndarray, b_woman: np.ndarray, db_woman: Union[np.ndarray, None],
                nb_woman: np.ndarray, sigma: float, tau: float) \
        -> Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray, Union[np.ndarray, None], np.ndarray]:
    """
    computes the hessians of :math:`H_y(V_{\\cdot y})` for one type of woman

    :param np.ndarray Vwoman: the values of :math:`V_{\\cdot y}`

    :param np.ndarray b_woman: the values of b for this type

    :param np.ndarray db_woman: the values of db for this type (if n_pars_b_women > 0)

    :param float nb_woman: the number of women of this type

    :param float sigma: the FC-MNL sigma

    :param float tau: the FC-MNL tau

    :return: a 5-uple with the value of :math:`H_y(V_{\\cdot y})` and its derivatives wrt the parameters of b, \
        and wrt :math:`V_{\\cdot y}`, and the hessians in (d,V) and (V,V)
    """
    resus_y = \
        fcmnl_generator(Vwoman, b_woman, sigma, tau, derivs=2)
    value_y = resus_y.value
    Hy_val = log(value_y) / tau + EULER_CONSTANT
    Hval = nb_woman * Hy_val

    fac_y = 1.0 / tau / value_y
    my_fy = nb_woman * fac_y
    fac_y2 = fac_y / value_y
    my_fy2 = nb_woman * fac_y2

    dgenfun_dV = resus_y.gradients[1]
    gradH_V = my_fy * dgenfun_dV
    d2genfun_dVV = resus_y.hessians[1]
    d2H_VV = my_fy * d2genfun_dVV - my_fy2 * np.outer(dgenfun_dV, dgenfun_dV)

    if db_woman is not None:
        dgenfun_db = resus_y.gradients[0]
        dgenfun_db_p = np.einsum('ijk, ij->k', db_woman, dgenfun_db)
        gradH_d = my_fy * dgenfun_db_p
        d2genfun_dbV = resus_y.hessians[0]
        d2genfun_dbV_p = np.einsum('ijk, ijl->kl', db_woman, d2genfun_dbV)
        d2H_dV = my_fy * d2genfun_dbV_p - my_fy2 * \
            np.outer(dgenfun_db_p, dgenfun_dV)
        return (Hval, gradH_d, gradH_V, d2H_dV, d2H_VV)
    else:
        return (Hval, None, gradH_V, None, d2H_VV)


def derivs_GplusH_fcmnl(U: np.ndarray, model_params: CupidParamsFcmnl, Phi: np.ndarray,
                        pars_b_men: np.ndarray, pars_b_women: np.ndarray,
                        derivs: int = 1) -> DerivsGHReturn:
    """
    evaluates values and derivatives of G_b(U) and H_b(Phi-U) for the FC-MNL model

    :param np.ndarray  U: (XY) array of men systematic utilities

    :param model_params: the model data

    :param Phi: the surplus matrix

    :param pars_b_men: the parameters of b for men

    :param pars_b_women: the parameters of b for women

    :return: the values of G_b and H_b

             if derivs >=1, (XY) arrays for:
               * the gradients of G in b and U
               * and of H in b and V

             and if derivs=2: G_bU, and H_bV (XY) arrays,  G_UU and H_VV both (XY, XY) arrays, \
             and the d2G(X,Y,Y) and d2H(Y,X,X)
    """
    men_margins, women_margins = model_params.men_margins, \
        model_params.women_margins
    ncat_men, ncat_women = men_margins.size, women_margins.size
    n_prod_categories = ncat_men * ncat_women
    make_b = model_params.make_b
    n_pars_b_men = model_params.n_pars_b_men
    n_pars_b_women = model_params.n_pars_b_women
    n_pars_b = n_pars_b_men + n_pars_b_women
    sigma = model_params.sigma
    tau = model_params.tau

    b_men, db_men, b_women, db_women = make_b(
        pars_b_men, pars_b_women, ncat_men, ncat_women)
    Umat = U.reshape((ncat_men, ncat_women))
    Vmat = Phi - Umat
    gradG_U = np.zeros_like(Umat)
    gradH_V = np.zeros_like(Umat)
    gradG_d = np.zeros(n_pars_b)
    gradH_d = np.zeros(n_pars_b)

    resus_ders = DerivsGHReturn()

    if derivs == 0:

        Gx_men = np.zeros(ncat_men)
        for iman in range(ncat_men):
            Gx_men[iman] = value_Gx(
                Umat[iman, :], b_men[iman, :, :], sigma, tau)

        Gval = np.sum(np.array(Gx_men * men_margins))

        Hy_women = np.zeros(ncat_women)
        for iwoman in range(ncat_women):
            Hy_women[iwoman] = value_Hy(
                Vmat[:, iwoman], b_women[iwoman, :, :], sigma, tau)

        Hval = np.sum(np.array(Hy_women * women_margins))

        resus_ders.values = [Gval, Hval]

    elif derivs == 1:

        if n_pars_b_men > 0:
            db_men_used = [db_men[iman, :, :, :] for iman in range(ncat_men)]
        else:
            db_men_used = [None] * ncat_men  # no derivatives
        Gval = 0.0
        for iman in range(ncat_men):
            grad_Gx_man = gradients_Gx(Umat[iman, :], b_men[iman, :, :],
                                       db_men_used[iman],
                                       men_margins[iman], sigma, tau)
            Gval += grad_Gx_man[0]
            gradG_U[iman, :] = grad_Gx_man[2]
            if n_pars_b_men > 0:
                gradG_d[:n_pars_b_men] += grad_Gx_man[1]

        if n_pars_b_women > 0:
            db_women_used = [db_women[iwoman, :, :, :] for iwoman in range(ncat_women)]
        else:
            db_women_used = [None] * ncat_women  # no derivatives
        Hval = 0.0
        for iwoman in range(ncat_women):
            grad_Hy_woman = gradients_Hy(Vmat[:, iwoman], b_women[iwoman, :, :],
                                         db_women_used[iwoman],
                                         women_margins[iwoman], sigma, tau)
            Hval += grad_Hy_woman[0]
            gradH_V[:, iwoman] = grad_Hy_woman[2]
            if n_pars_b_women > 0:
                gradH_d[n_pars_b_men:n_pars_b] += grad_Hy_woman[1]

        resus_ders.values = [Gval, Hval]

        resus_ders.gradients = [gradG_d, gradG_U,
                                gradH_d, gradH_V]

    elif derivs == 2:
        d2G_dU = np.zeros((n_pars_b, n_prod_categories))
        d2G_UU = np.zeros((n_prod_categories, n_prod_categories))
        d2H_dV = np.zeros((n_pars_b, n_prod_categories))
        d2H_VV = np.zeros((n_prod_categories, n_prod_categories))

        if n_pars_b_men > 0:
            db_men_used = [db_men[iman, :, :, :] for iman in range(ncat_men)]
        else:
            db_men_used = [None] * ncat_men  # no derivatives

        Gval = 0.0
        ivar = 0
        for iman in range(ncat_men):
            hess_Gx_man = hessians_Gx(Umat[iman, :], b_men[iman, :, :],
                                          db_men_used[iman], men_margins[iman], sigma, tau)
            slice_man = slice(ivar, ivar + ncat_women)
            Gval += hess_Gx_man[0]
            gradG_U[iman, :] = hess_Gx_man[2]
            d2G_UU[slice_man, slice_man] = hess_Gx_man[4]
            if n_pars_b_men > 0:
                gradG_d[:n_pars_b_men] += hess_Gx_man[1]
                d2G_dU[:n_pars_b_men, slice_man] = hess_Gx_man[3]
            ivar += ncat_women

        if n_pars_b_women > 0:
            db_women_used = [db_women[iwoman, :, :, :] for iwoman in range(ncat_women)]
        else:
            db_women_used = [None] * ncat_women  # no derivatives

        Hval = 0.0
        for iwoman in range(ncat_women):
            hess_Hy_woman = hessians_Hy(Vmat[:, iwoman], b_women[iwoman, :, :],
                                            db_women_used[iwoman], women_margins[iwoman], sigma, tau)
            slice_woman = slice(iwoman, n_prod_categories, ncat_women)
            Hval += hess_Hy_woman[0]
            gradH_V[:, iwoman] = hess_Hy_woman[2]
            d2H_VV[slice_woman, slice_woman] = hess_Hy_woman[4]
            if n_pars_b_women > 0:
                gradH_d[n_pars_b_men:n_pars_b] += hess_Hy_woman[1]
                d2H_dV[n_pars_b_men:n_pars_b, slice_woman] = hess_Hy_woman[3]

        resus_ders.values = [Gval, Hval]

        resus_ders.gradients = [gradG_d, gradG_U,
                                gradH_d, gradH_V]
        resus_ders.hessians = [d2G_dU, d2G_UU, d2H_dV, d2H_VV]

    else:
        bs_error_abort(f"derivs_GplusH_fcmnl: derivs={derivs} is illegal")

    return resus_ders


def grad_GplusH_fcmnl(U: np.ndarray,
                      args: Tuple[CupidParamsFcmnl, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """

    :param U: the utilities of men

    :param args: (model_params_fcmnl, Phi, pars_b_men, pars_b_women)

    :return: the gradient of :math:`G_b(U)+H_b(\\Phi-U)` wrt :math:`U`
    """
    model_params, Phi, pars_b_men, pars_b_women = args
    resus_ders = derivs_GplusH_fcmnl(
        U, model_params, Phi, pars_b_men, pars_b_women)
    gradients = resus_ders.gradients
    gradG_U, gradH_V = gradients[1], gradients[3]
    n_prod_categories = gradG_U.size
    return (gradG_U - gradH_V).reshape(n_prod_categories)


if __name__ == "__main__":

    make_b = make_b8
    pars_b_women = np.array([0.1, 0.1])
    pars_b_men = np.array([0.1, 0.1])

    # we generate a Choo and Siow homo matching
    ncat_men = ncat_women = 5
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
            bases_surplus[ix, iy, 3] = (x_men[ix] - y_women[iy]) \
                * (x_men[ix] - y_women[iy])

    men_margins = np.random.uniform(1.0, 10.0, size=ncat_men)
    women_margins = np.random.uniform(1.0, 10.0, size=ncat_women)

    # np.random.normal(mu, sigma, size=n_bases)
    true_surplus_params = np.array([3.0, -1.0, -1.0, -2.0])
    Phi = bases_surplus @ true_surplus_params

    sigma = 0.5
    tau = 1.1

    n_pars_b_men = pars_b_men.size
    n_pars_b_women = pars_b_women.size
    n_pars_b = n_pars_b_men + n_pars_b_women

    model_params_fcmnl = CupidParamsFcmnl(men_margins=men_margins, women_margins=women_margins,
                                          bases_surplus=bases_surplus,
                                          make_b=make_b,
                                          sigma=sigma, tau=tau,
                                          n_pars_b_men=n_pars_b_men, n_pars_b_women=n_pars_b_women)

    U_init_mat = Phi / 34.0
    U_init = U_init_mat.reshape(n_prod_categories)

    man0 = 3
    woman0 = 2

    print_stars("First we check the fcmnl_generator function")

    EPS = GRADIENT_STEP

    b_men, db_men, b_women, db_women = make_b(
        pars_b_men, pars_b_women, ncat_men, ncat_women)
    b_man0 = b_men[man0, :, :]
    U_man0 = U_init_mat[man0, :]
    resus0 = fcmnl_generator(U_man0, b_man0, sigma, tau, derivs=2)

    grad_b = resus0.gradients[0]
    grad_U = resus0.gradients[1]
    hess_bU = resus0.hessians[0]
    hess_UU = resus0.hessians[1]

    grad_b_num = np.empty((ncat_women, ncat_women))
    grad_U_num = np.empty(ncat_women)

    for iwoman1 in range(ncat_women):
        print(f"    iwoman1={iwoman1}")
        for iwoman2 in range(ncat_women):
            b_man1 = b_man0.copy()
            b_man1[iwoman1, iwoman2] += EPS
            resus1 = fcmnl_generator(U_man0, b_man1, sigma, tau, derivs=None)
            grad_b_num[iwoman1, iwoman2] = (resus1.value - resus0.value) / EPS

    error_grad_b = (grad_b_num - grad_b)/grad_b_num
    describe_array(error_grad_b, "error grad_b")

    for iwoman in range(ncat_women):
        U_man1 = U_man0.copy()
        U_man1[iwoman] += EPS
        resus1 = fcmnl_generator(U_man1, b_man0, sigma, tau, derivs=None)
        grad_U_num[iwoman] = (resus1.value - resus0.value) / EPS

    error_grad_U = (grad_U_num - grad_U)/grad_U_num
    describe_array(error_grad_U, "error grad_U")

    hess_bU_num = np.empty((ncat_women, ncat_women, ncat_women))
    hess_UU_num = np.empty((ncat_women, ncat_women))

    for iwoman in range(ncat_women):
        U_man1 = U_man0.copy()
        U_man1[iwoman] += EPS
        resus1 = fcmnl_generator(U_man1, b_man0, sigma, tau, derivs=1)
        hess_UU_num[iwoman, :] = (
            resus1.gradients[1] - resus0.gradients[1]) / EPS

    error_hess_UU = hess_UU_num - hess_UU
    describe_array(error_hess_UU, "error hess_UU")

    for iwoman1 in range(ncat_women):
        for iwoman2 in range(ncat_women):
            b_man1 = b_man0.copy()
            b_man1[iwoman1, iwoman2] += EPS
            resus1 = fcmnl_generator(U_man0, b_man1, sigma, tau, derivs=1)
            hess_bU_num[iwoman1, iwoman2, :] = (
                resus1.gradients[1] - resus0.gradients[1]) / EPS

    error_hess_bU = hess_bU_num - hess_bU
    describe_array(error_hess_bU, "error hess_bU")

    print_stars("now we check the derivs function")
    res_ders0 = derivs_GplusH_fcmnl(U_init, model_params_fcmnl, Phi, pars_b_men, pars_b_women,
                                    derivs=2)
    gradG_d, gradG_U, gradH_d, gradH_V = res_ders0.gradients
    d2G_dU, d2G_UU, d2H_dV, d2H_VV = res_ders0.hessians

    d2G_dU_num = np.zeros((n_pars_b, n_prod_categories))
    d2G_UU_num = np.zeros((n_prod_categories, n_prod_categories))
    i = 0
    for iman in range(ncat_men):
        print(f"    iman = {iman}")
        for iwoman in range(ncat_women):
            U_init1 = U_init.copy()
            U_init1[i] += EPS
            res_ders1 = derivs_GplusH_fcmnl(U_init1, model_params_fcmnl, Phi, pars_b_men, pars_b_women,
                                            derivs=1)
            gradG_d1 = res_ders1.gradients[0]
            gradG_U1 = res_ders1.gradients[1]
            d2G_dU_num[:, i] = (gradG_d1 - gradG_d) / EPS
            d2G_UU_num[:, i] = (
                gradG_U1 - gradG_U).reshape(n_prod_categories) / EPS
            i += 1

    error_d2G_dU = d2G_dU_num - d2G_dU
    describe_array(error_d2G_dU, "error d2G_dU")

    error_d2G_UU = d2G_UU_num - d2G_UU
    describe_array(error_d2G_UU, "error d2G_UU")

    d2H_dV_num = np.zeros((n_pars_b, n_prod_categories))
    d2H_VV_num = np.zeros((n_prod_categories, n_prod_categories))
    i = 0
    for iman in range(ncat_men):
        print(f"    iman={iman}")
        for iwoman in range(ncat_women):
            U_init1 = U_init.copy()
            U_init1[i] += EPS
            res_ders1 = derivs_GplusH_fcmnl(U_init1, model_params_fcmnl, Phi, pars_b_men, pars_b_women,
                                            derivs=1)
            gradH_d1 = res_ders1.gradients[2]
            gradH_V1 = res_ders1.gradients[3]
            d2H_dV_num[:, i] = -(gradH_d1 - gradH_d) / EPS
            d2H_VV_num[:, i] = - \
                (gradH_V1 - gradH_V).reshape(n_prod_categories) / EPS
            i += 1

    error_d2H_dV = d2H_dV_num - d2H_dV
    describe_array(error_d2H_dV, "error d2H_dV")

    error_d2H_VV = d2H_VV_num - d2H_VV
    describe_array(error_d2H_VV, "error d2H_VV")
