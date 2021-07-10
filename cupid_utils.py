from math import sqrt
import numpy as np
import scipy.stats as sts
import sys
from typing import Optional
from traceback import extract_stack

root_dir = "./"

EULER_CONSTANT = 0.5772156649

"""
numbers of households, in the population and in the sample
"""
N_HOUSEHOLDS_POP = 13_272_301
N_HOUSEHOLDS_OBS = 224_068

MIN_MUS = 1  # use larger values in bad cases, to bound below the mus
MIN_MUS_NORM = 1e-8  # use larger values in bad cases, to bound below the normalized mus

GRADIENT_STEP = 1e-6  # to compute gradients numerically


def print_stars(title: Optional[str] = None, n: Optional[int] = 70) -> None:
    """
    prints a title within stars

    :param str title:  title

    :param int n: number of stars on line

    :return: prints a starred line, or two around the title
    """
    line_stars = '*' * n
    print()
    print(line_stars)
    if title:
        print(title.center(n))
        print(line_stars)
    print()


def bs_name_func(back: Optional[int] = 2) -> str:
    """
    get the name of the current function, or further back in stack

    :param int back: 2 is current function, 3 the function that called it etc

    :return: a string
    """
    stack = extract_stack()
    filename, codeline, funcName, text = stack[-back]
    return funcName


def bs_error_abort(msg: Optional[str] = "error, aborting") -> None:
    """
    report error and abort

    :param str msg: a message

    :return: exit with code 1
    """
    print_stars(f"{bs_name_func(3)}: {msg}")
    sys.exit(1)


def describe_array(v: np.ndarray, name: str = "v") -> None:
    """
    descriptive statistics on an array interpreted as a vector

    :param np.array v: the array

    :param str name: its name

    :return: nothing
    """
    print_stars(f"{name} has:")
    d = sts.describe(v, None)
    print(f"Number of elements: {d.nobs}")
    print(f"Minimum: {d.minmax[0]}")
    print(f"Maximum: {d.minmax[1]}")
    print(f"Mean: {d.mean}")
    print(f"Stderr: {sqrt(d.variance)}")
    return


def eval_moments(matching_patterns: np.ndarray, bases_surplus: np.ndarray) -> np.ndarray:
    """
    computes comoments

    :param matching_patterns: muxy matrix

    :param bases_surplus: bases

    :return: values of comoments
    """
    return np.einsum('ij,ijk->k', matching_patterns, bases_surplus)

