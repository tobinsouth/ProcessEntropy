import numba
from numba import jit, prange
import numpy as np
import math
import nltk

from ProcessEntropy.Preprocessing import *

@jit(nopython = True)
def get_all_self_lambdas(source, lambdas):
    """ 
    Finds all the the shortest subsequences of the target, 
    that are contained in the subsequence of the source,
    with the source cutoff at the location set in relative_pos.
    
    See function find_lambda_jit for description of 
        Lambda_i(target|source)
    
    Args:        
        source: Arry of ints, usually corresponding to hashed words.
            
        lambdas: A premade array of length(target), usually filled with zeros. 
            Used for efficiency reasons.
        
    Return:
        A list of ints, denoting the value for Lambda for each index in the target. 
    
    """
    
    N = len(source)
    
    for i in prange(1, N): 
    
        # The target process is everything ahead of i.
        t_max = 0
        c_max = 0

        for j in range(0, i): # Look back at the past
            if source[j] == source[i]: # Check if matches future's next element
                c_max = 1
                for k in range(1,min(N-i, i)): # Look through more of future
                    if source[j+k] != source[i+k]:
                        break
                    else:
                        c_max = c_max+1

                if c_max > t_max:
                    t_max = c_max 

        lambdas[i] = t_max+1
            
    return lambdas



def self_entropy_rate(source, get_lambdas = False):
    """
    Args:
        source: The source is an array of ints.
    Returns: 
        The non-parametric estimate of the entropy rate based on match lengths.
        
        
    $$
    \hat{h}(S)=\frac{N \log _{2} N{\sum_{i=1}^{N \Lambda_{i}(S)}
    $$
    
    This is described mathematically in [1] as,
     
    [1] I. Kontoyiannis, P.H. Algoet, Yu.M. Suhov, and A.J. Wyner. Nonparametric entropy 
    esti mation for stationary processes and random fields, with applications to English text. 
    IEEE Transactions on Information Theory, 44(3):1319–1327, May 1998.
        
    """
    
    N = len(source)
    source = np.array(source)
    lambdas = np.zeros(N)
    lambdas = get_all_self_lambdas(source, lambdas)
    
    if get_lambdas:
        return lambdas
    else:
        return N*math.log(N,2) / np.sum(lambdas)


        