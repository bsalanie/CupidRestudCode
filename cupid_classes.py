""" some classes used throughout """

from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple, Union
import numpy as np


@dataclass
class MatchingMus:
    """
    matching patterns (or anything structured like them)
    """
    muxy: np.ndarray
    mux0: np.ndarray
    mu0y: np.ndarray

    def unpack(self):
        return self.muxy, self.mux0, self.mu0y


@dataclass
class CupidParams:
    """
    describes the model we deal with
    """
    men_margins: np.ndarray
    women_margins: np.ndarray
    bases_surplus: np.ndarray
    mus_and_maybe_grad: Optional[Callable] = None
    observed_matching: Optional[MatchingMus] = None
    ipfp_solver: Optional[Callable] = None


@dataclass
class CupidParamsCSHeteroxy(CupidParams):
    """
    additional parameters for the CSHXY model
    """
    covariates_sigma: Optional[np.ndarray] = None
    covariates_tau: Optional[np.ndarray] = None


@dataclass
class CupidParamsFcmnl(CupidParams):
    """
    additional parameters for the FC-MNL model; defaults as in Davis and Schiraldi
    """
    tol_agd: Optional[float] = 1e-9
    sigma: Optional[float] = 0.5
    tau: Optional[float] = 1.1
    n_pars_b_men: Optional[int] = None
    n_pars_b_women: Optional[int] = None
    make_b: Optional[Callable] = None


# the type of the model parameters classes
ModelParams = Union[CupidParams, CupidParamsCSHeteroxy, CupidParamsFcmnl]


@dataclass
class GHReturn:
    """
    object we return from the evaluation of :math:`G` and :math:`H`
    """
    value: Optional[float] = None
    derivs: Optional[int] = None
    gradients: Optional[Tuple[np.ndarray]] = None
    hessians: Optional[Tuple[np.ndarray]] = None


@dataclass
class DerivsGHReturn:
    """
    object we return from the evaluation of the derivatives of :math:`G` and :math:`H`
    """
    values: Optional[List[float]] = None
    gradients: Optional[List[np.ndarray]] = None
    hessians: Optional[List[np.ndarray]] = None
