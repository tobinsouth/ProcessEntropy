import numpy as np
import re

from nltk.tokenize import TweetTokenizer # This will be our chosen tokenzier for cleaning text before hashing.
import re
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False) 


def fnv(data):
    """
    Fowler–Noll–Vo hash algorithm. 
    
    This algorithm is an efficient non-cryptographic hash function that is ideal 
    for hashing strings in such a way that speed is prioritized while minimising 
    collisions. Please node that on large vocab sets, collisions may occur.
    
    Args:
        data: byte type array or string
    
    Return: 
        32 bit hash value 
    
    See:
        https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
        
    Credit:
        https://github.com/znerol/py-fnvhash/blob/master/fnvhash/__init__.py
    """
    if not isinstance(data, bytes):
        data = str.encode(data)

    hval = 0
    for byte in data:
        hval = (hval * 0x100000001b3) % 2**32
        hval = hval ^ byte
    return hval


def custom_tokenize(text, basic_tokenize = False):
    """ Tokenizer to extract the key words and numbers. 
    Args:
        text: A long string, usually a tweet.
        basic_tokenize: Setting true will use only the nltk TweetTokenizer. 
            Otherwise all non-whitespace characters will be removed as well as urls.
    Return:
        An array of tokens as strings.
    
    Process:
        - Use tweet tokenizer (semi smart about breaking things up and keep emoji’s
        - Lowercase everything
        - Shorten lengths ‘yessss’ → ‘yess’  (built in functionality) 
        - Remove URL’s & pic.twitter (Removes 70% of the set of unique tokens)
        - Remove all not alphanumeric characters  (Removes 17% of all tokens )
        
    """
    split = tknzr.tokenize(text)
    if not basic_tokenize:
        clean_split = []
        for w in split:
            if 'http' not in w:
                if 'pic.twitter' not in w:
                    wa = w.split('-')
                    wa = [re.sub(r'\W+', '', w) for w in wa]
                    wa = [w for w in wa if w != ''] # Removes 17% of tokens, beware
                    clean_split.extend(wa)
        return clean_split
    else:
        return split

def tweet_to_hash_array(text):
    """
    Takes a single string of text (e.g. a single tweet).
    Tokenizes the string, removing twitter handles, and reducing length, 
    (e.g. trueeeee becomes truee).
    Uses Fowler–Noll–Vo hash function on tokens to give digestable numbers.
    """
    if type(text) is not str: 
        return text
    try:
        return [fnv(str.encode(w)) for w in custom_tokenize(text)]
    
    except UnicodeEncodeError: # This is to deal with encoding unicode surrogates
        encoded = []
        for w in tknzr.tokenize(text):
            try:
                encoded.append(fnv(str.encode(w)))
            except UnicodeEncodeError:
                continue # They're simply ignored
        return encoded




