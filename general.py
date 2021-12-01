from matplotlib import pyplot as plt
import numpy as np
from numpy import pi
from copy import deepcopy

from sys import path
from os import environ

from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)


# General parameters:
eta = 1e-8
NUM_SPINS = 2

# General helper functions:


class Infix:

    def __init__(self, function):
        self.function = function

    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))

    def __or__(self, other):
        return self.function(other)

    def __rlshift__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))

    def __rshift__(self, other):
        return self.function(other)

    def __call__(self, value1, value2):
        return self.function(value1, value2)


def fermi(beta, xi):
    return 1.0 / (np.exp(beta * xi) + 1.0)


def dagger(A):
    return np.conj(np.transpose(A))


def nearly_equal(a, b, sig_fig=9):
    return (a == b or np.abs(a - b) < 0.1 ** sig_fig)
eqls = Infix(nearly_equal)


class Struct:

    def __init__(self, **entries):
        self.__dict__.update(entries)
