from numba import jit, prange
import numpy as np
import warnings
from tqdm import tqdm
import LCSFinder as lcs

from ProcessEntropy.Preprocessing import *


def find_lambda(target, source):
    """
    Finds the longest subsequence of the target array, 
    starting from index 0, that is contained in the source array.
    Returns the length of that subsequence + 1.
    
    i.e. returns the length of the shortest subsequence starting at 0 
    that has not previously appeared.
    
    Args:
        target: NumPy array, preferable of type int.
        source: NumPy array, preferable of type int.
    
    Returns:
        Integer of the length.
        
    """
    
    if (len(target)==0) or (len(source)==0):
        return 0


    # set up objects
    target = np.array(target, dtype = np.uint32)
    source = np.array(source, dtype = np.uint32)

    source = lcs.Vector1D([int(x) for x in ([np.floor(x) for x in source])])
    target = lcs.Vector1D([int(x) for x in ([np.floor(x) for x in target])])

    ob = lcs.LCSFinder(target,source)

    # set up indexes
    l_t = lcs.Vector2D(tuple((0,len(source)) for _ in range(1)))

    # calculate longest match length
    t_max = ob.ComputeAllLCSs(l_t)[0]
                
    return t_max+1



def get_all_lambdas(target, source, relative_pos):
    """ 
    Finds all the the longest subsequences of the target, 
    that are contained in the sequence of the source,
    with the source cut-off at the location set in relative_pos.
    
    See function find_lambda for description of 
        Lambda_i(target|source)
    
    Args:
        target: Array of ints, usually corresponding to hashed words.
        
        source: Array of ints, usually corresponding to hashed words.
        
        relative_pos: list of integers with the same length as target denoting the                                                               
            relative time ordering of target vs. source. These integers tell us the 
            position relative_pos[x] = i in source such that all symbols in source[:i] 
            occurred before the x-th word in target.  
            
        lambdas: A pre-made array of length(target), usually filled with zeros. 
            Used for efficiency reasons.
        
    Return:
        A list of ints, denoting the value for Lambda for each index in the target. 
    
    Added check that if relative_pos = 0 (we have no source, we return lambda = 0)
    Note! We don't check if i is longer than target.
    """

    # set up the objects we need
    target = np.array(target, dtype = np.uint32)
    source = np.array(source, dtype = np.uint32)
    relative_pos = np.array(relative_pos, dtype = np.uint32)
    source = lcs.Vector1D([int(x) for x in ([np.floor(x) for x in source])])
    target = lcs.Vector1D([int(x) for x in ([np.floor(x) for x in target])])

    ob = lcs.LCSFinder(target,source)

    l_t =  lcs.Vector2D(tuple((i,int(relative_pos[i])) for i in range(len(relative_pos)) if int(relative_pos[i])!=0))

    return np.array([x+1 for x in ob.ComputeAllLCSs(l_t)])


def timeseries_cross_entropy(time_tweets_target, time_tweets_source, please_sanitize = True, get_lambdas = False):
    """ 
    Finds the cross entropy H_cross(target|source) where the processes are embedded in time.
    
    i.e. How many bits we would need to encode the data in target
    using information in the source that is before the current time. 
    
    
    This is described mathematically in [1] as,

    T = target
    S = source
    
    $$
    \hat{h}_{ \times}(T | S)=\frac{N_{T} \log _{2} N_{S}}{\sum_{i=1}^{N_{T}} \Lambda_{i}(T | S)}
    $$

    
    Args:
        time_tweets_target: A list of tuples with (time, tweet_content). 
            This is the stream of new information that we can testing the ability to encode.
            If please_sanitize = True (default) then tweet_content can be a string.
    
        time_tweets_source: A list of tuples with (time, tweet_content). 
            This is the stream of previous information from which we try to encode the target.
            
        please_sanitize: Option to have the tweet string converted to numpy int arrays for speed.
            If False, please sanitize tweet into a list of tokenized words, ideally converting these to ints,
            via a hash.
            
        get_lambdas: Boolean choice to return the list of all calculated Lambda values for each point 
            in target. Usually used for debugging.
    
    Return: 
        The cross entropy as a float
    
    
    [1] I. Kontoyiannis, P.H. Algoet, Yu.M. Suhov, and A.J. Wyner. Nonparametric entropy 
    estimation for stationary processes and random fields, with applications to English text. 
    IEEE Transactions on Information Theory, May 1998.

    Credit to Bagrow and Mitchell for code ideas that I've stolen for this function. 
       
    """
    
    # Decorate tweets (so we can distinguish the users), before sorting in time:     
    
    if please_sanitize: # Option to have the tweet string converted to numpy int arrays for speed.
        # tweet_to_hash_array function can be found in package.
        decorated_target = [ (time,"target",tweet_to_hash_array(tweet)) for time,tweet in time_tweets_target ]
        decorated_source = [ (time,"source",tweet_to_hash_array(tweet)) for time,tweet in time_tweets_source ]
    else:
        decorated_target = [ (time,"target",tweet) for time,tweet in time_tweets_target ]
        decorated_source = [ (time,"source",tweet) for time,tweet in time_tweets_source ]
        
    # Join time series:
    time_tweets = decorated_target + decorated_source

    # Sort in place by time:                                                                                                                 
    time_tweets.sort()

    # Loop over combined tweets and build word vectors and target->source relative_pos:                                                                     
    target, source, relative_pos = [], [], []
    for time,user,tweet in time_tweets:
        words = tweet
        if user == "target":
            target.extend(words)
            relative_pos.extend( [len(source)]*len(words) )
        else:                                                                                                                        
            source.extend(words)
    
    # get lambdas
    lambdas = get_all_lambdas(target, source, relative_pos)
    
    if get_lambdas:
        return lambdas
    return  len(target)*np.log2(len(source)) / np.sum(lambdas)


def conditional_entropy(target, source):
    """
    Finds the simple conditional entropy as a process.

    Entropy of target process conditional on full knowledge of states of source process.

    Args:
    target: A 1-D numpy array of integers.

        This is the stream of new information that we can testing the ability to encode.

    source: A 1-D numpy array of integers.
            This is the stream of previous information from which we try to encode the target.

       get_lambdas: Boolean choice to return the list of all calculated Lambda values for each point 
            in target. Usually used for debugging.

    Return: 
        The conditional entropy as a float
    """
    lambdas = np.zeros(len(target))
    for i in prange(0, len(target)):
        lambdas[i] = find_lambda(target[i:], source) 
            
	# Previously we allowed the return of lambdas but this significantly slows down the code.
    return len(target)*np.log2(len(source)) / np.sum(lambdas)


def cross_shannon(target, source):
    """A function to compute the cross shannon entropy from a source to a target. This uses the probability distribution (in the target and source) of the values that exists in the intersection of the state space.

    $$\sum_x p(x) log q(x)$$

    where q is the pdf of the target and p is the pdf of the source.

    Args:
        target (array like): Ideally an np.array of ints for speed purposes. Should work with a list and when elements are strings.
        source (array like): Same as target.

    Returns:
        int: Entropy value 
    """
    tvalues, tcounts = np.unique(target, return_counts=True)
    svalues, scounts = np.unique(source, return_counts=True)
    tprobs = tcounts / len(target) # Maybe these should be divided the length of the intersection
    sprobs = scounts / len(source)
    entropy = 0
    for i, v in enumerate(tvalues):
        p_t = tprobs[i]
        s_index = np.where(svalues == v)[0]
        if len(s_index) > 0:
            p_s = sprobs[s_index[0]]
            entropy += np.log2(p_t)*p_s
    return -entropy

def pairwise_information_flow(data, text_col = 'tweet', label_col = 'username', time_col=None, network_method = True, show_warnings = True, progressbar = True, return_entropies = True):
    """
    A function to compute the pairwise information flow between the text in text_col for each text producer in in label_column.

    Args:
        data (pandas dataframe): The dataframe in long format with rows containing (text producer, text, [time of text]) .
        text_col (str): The name of the column to use as the text.
        label_col (str): The name of the column to groupby as the text producer. If set to 'index' the index column will be used.
        time_col (str, None): The name of the column with the times each text was produced. If None then a non-time-synced entropy will be used.

    Returns:
        pandas dataframe: The pairwise directed information flow between source and target pairs, as well as intermediate results.
    """
    if not time_col:
        import itertools
    import pandas as pd

    # Create time_tweets for each producer 
    all_time_tweets = {}
    for producer, group_df in data.groupby(label_col):
        # A list of tuples with (time, tweet_content).  `tweet_to_hash_array` function can be found in package.
        all_time_tweets[producer] = [(time, tweet_to_hash_array(tweet)) for time, tweet in zip(group_df[time_col] if time_col else [0]*len(group_df), group_df[text_col])]

        if show_warnings:
            total_number_of_tokens = sum([len(tweet) for _, tweet in all_time_tweets[producer]])
            if total_number_of_tokens < 1000:
                warnings.warn(("Text producer %d has less than 1000 tokens, entropy estimation may not converge for" % producer) + "\nYou can confirm this with `ProcessEntropy.SelfEntropy.convergence`." )

    if show_warnings and len(all_time_tweets) < 2:
        raise ValueError("Only one text producer was found, there are no pairs to compute.")

    # Calculate pairwise entropies
    all_producers = list(all_time_tweets.keys())
    all_entropy_results = {}
    for producer_source in tqdm(all_producers,disable=(not progressbar)):
        for producer_target in all_producers:
            if producer_source != producer_target:
                time_tweets_source = all_time_tweets[producer_source]
                time_tweets_target = all_time_tweets[producer_target]
                if time_col:
                    cross_entropy = timeseries_cross_entropy(time_tweets_target, time_tweets_source, please_sanitize = False)                            
                else: # If time is not present, a simple conditional entropy will be used.
                    tweets_target = np.array(list(itertools.chain.from_iterable([tweet for _, tweet in time_tweets_target])))
                    tweets_source = np.array(list(itertools.chain.from_iterable([tweet for _, tweet in time_tweets_source])))
                    cross_entropy = conditional_entropy(tweets_target, tweets_source)

                all_entropy_results[(producer_source, producer_target)] = cross_entropy
        
    # If we are not using the network method, we need to calculate self entropy rates
    if show_warnings and (len(all_producers) < 4):
        warnings.warn('Not enough users to normalize by network neighborhood. Normalising by self entropy rate.')
        network_method = False

    if not network_method:
        from ProcessEntropy.SelfEntropy import tweet_self_entropy
        self_entropy_results = {producer: tweet_self_entropy(time_tweets) for producer, time_tweets in all_time_tweets.items()}

    # Create a dataframe with the results
    results = []
    for producer_source in  all_producers:
        for producer_target in all_producers:
            if producer_source != producer_target:
                hStoT = all_entropy_results[(producer_source, producer_target)]
                hTtoS = all_entropy_results[(producer_target, producer_source)]
                if network_method:
                    hXtoS = np.mean([all_entropy_results[(X, producer_source)] for X in all_producers if X != producer_source])
                    hXtoT = np.mean([all_entropy_results[(X, producer_target)] for X in all_producers if X != producer_target])
                    flow = (hStoT / hXtoT) - (hTtoS / hXtoS)
                    results.append([producer_source, producer_target, flow] + ([hStoT, hTtoS] if return_entropies else []))
                else:
                    hT = self_entropy_results[producer_target]
                    hS = self_entropy_results[producer_source]
                    flow = (hStoT / hT) - (hTtoS / hS)
                    results.append([producer_source, producer_target, flow] + ([hStoT, hTtoS, hT, hS] if return_entropies else []))
    if network_method:
        results = pd.DataFrame(results, columns = ['source', 'target', 'flow'] + (['entropyStoT', 'entropyTtoS'] if return_entropies else []))
    else:
        results = pd.DataFrame(results, columns = ['source', 'target', 'flow'] + (['entropyStoT', 'entropyTtoS', 'selfEntropyT', 'selfEntropyS'] if return_entropies else []))
    return results