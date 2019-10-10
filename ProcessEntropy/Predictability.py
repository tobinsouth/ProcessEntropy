# This is a bonus file to help convert to predictabilties.

from scipy.optimize import fsolve
import numpy as np
import math

from ProcessEntropy.SelfEntropy import *
from ProcessEntropy.CrossEntropy import *

def predictability(S,N, inital_guess = 0.5): 
    """Finds the value of the predicatbility for a process with an entropy rate S and a vocabular size N."""
    # explodes for small values of N or large values of S :(
    try:
        f = lambda Pi : S + Pi*math.log(Pi,2) + (1 - Pi)*math.log(1 - Pi,2) - (1 - Pi)*math.log(N-1,2)
        PiMax = fsolve(f,inital_guess) 
    except:
        PiMax = 0
    return float(PiMax)


def process_predictability(process):
    """Calculates the predictability of the process. """
    entropy = nonparametric_entropy_estimate(process)
    N = len(set(process))
    return calc_predictability(entropy,N)



def cross_predictability(target,source):
    """Calculates the predictability of the target given the information in the source."""
    cross_entropy = timeseries_cross_entropy(target,source)
    N = len(set(target)) # THIS IS WHERE I"M NOT SURE WHAT N TO USE
    return predictability(entropy,N)


def surprise(probability):
    """Returns surprise value for given probability"""
    return log(1/probability,2)

