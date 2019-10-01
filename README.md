# ProcessEntropy

A toolkit for calculating process entropy quickly. With specific applications to tweets.


## Example Usage

```
# Load in tweets between 2018/11/16 to 2019/01/01
import pandas as pd
with open("example_data/BBCWorld_Tweets_small.csv", 'r') as f:
    BBC = pd.read_csv(f)
    
with open("example_data/BuzzFeedNews_Tweets_small.csv", 'r') as f:
    BuzzFeed = pd.read_csv(f)


# Find process entropy of BuzzFeed tweets
from ProcessEntropy.CrossEntropy import tweet_self_entropy

print(tweet_self_entropy(BuzzFeed['tweet']))


# Find cross entropy between BuzzFeed and BBC World
from ProcessEntropy.CrossEntropy import timeseries_cross_entropy

target = list(zip(BuzzFeed['created_at'], BuzzFeed['tweet']))
source = list(zip(BBC['created_at'], BBC['tweet']))

print(timeseries_cross_entropy(target, source))



```

## Requirements

- Python 3.x with packages:
	- Numba
	- NTLK
	- Numpy


## Installation

```
pip install ProcessEntropy
```
