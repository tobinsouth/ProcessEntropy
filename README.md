# ProcessEntropy

A toolkit for calculating sequence entropy rates quickly. Especially useful for cross entropy rates and measuing information flow. Application is aimed at tweets but can be used on an text or sequence like data.

## Example Usage

```
# Load in example tweets dataframe
import pandas as pd
example_tweet_data = pd.read_csv('example_data/example_tweet_data.csv')

import sys; sys.path.insert(0, 'ProcessEntropy')
from CrossEntropy import pairwise_information_flow

# Calculate information flow between users based on temporal text usage 
pairwise_information_flow(example_tweet_data, text_col = 'tweet', label_col = 'username', time_col = 'created_at')
```

## Requirements

- Python 3.x with packages:
	- numba
	- nltk (for tokenization)
	- numpy

## Installation

```
pip install ProcessEntropy
```
