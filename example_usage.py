# Example usage

from ProcessEntropy.SelfEntropy import *
from ProcessEntropy.CrossEntropy import *
from ProcessEntropy.Predictability import *


# Basic Usage
import nltk

emma = nltk.corpus.gutenberg.words('austen-emma.txt')
sense = nltk.corpus.gutenberg.words('austen-sense.txt')
shakespear = nltk.corpus.gutenberg.words('shakespeare-caesar.txt')

print("Legth of Emma:",len(emma))


emma_p  = np.array([fnv(word)  for word in emma]) 
sense_p  = np.array([fnv(word)  for word in sense])
shakespear_p  = np.array([fnv(word)  for word in shakespear])

print("Entropy of Emma (Novel) by Jane Austin is %.2f" % self_entropy_rate(emma_p))

# So in here I would normally have code to calculate the cross entropy between
# two sources without time series data. But lets call this a V2 product.


emma_predictability = predictability(self_entropy_rate(emma_p), len(set(emma_p)))
print("The predictability of Emma by Jane Austin is, %.2f"% emma_predictability)



# Twitter Usage

# Load in tweets between 2018/11/16 to 2019/01/01
import pandas as pd
with open("example_data/BBCWorld_Tweets_small.csv", 'r') as f:
    BBC = pd.read_csv(f)
    
with open("example_data/BuzzFeedNews_Tweets_small.csv", 'r') as f:
    BuzzFeed = pd.read_csv(f)

A = list(zip(BuzzFeed['created_at'], BuzzFeed['tweet']))
B = list(zip(BBC['created_at'], BBC['tweet']))

print("Cross Entropy:", timeseries_cross_entropy(A, B))

print("Buzzfeed Entropy:", tweet_self_entropy(BuzzFeed['tweet']))


## If you want to speed up processing, e.g. when you are going to run a user multiple times,
## you can pre-hash the strings.
# BuzzFeed['sanitized_tweet'] = pd.Series([tweet_to_hash_array(t) for t in BuzzFeed['tweet']])
# BBC['sanitized_tweet'] = pd.Series([tweet_to_hash_array(t) for t in BBC['tweet']])

# A = list(zip(BuzzFeed['created_at'], BuzzFeed['sanitized_tweet']))
# B = list(zip(BBC['created_at'], BBC['sanitized_tweet']))

# result = timeseries_cross_entropy(A, B, please_sanitize=False)

