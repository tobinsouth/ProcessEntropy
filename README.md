# ProcessEntropy

A toolkit for calculating sequence entropy rates quickly. Especially useful for cross entropy rates and measuring information flow. Application is aimed at tweets but can be used on text or sequence like data.

This toolkit uses a non-parametric entropy estimation technique which computes the longest match length between sequences to estimate their entropy. This functionality is provided by the LCSFinder package which calculates the longest common substrings with a fixed starting location of one substring. This algorithm employs properties of a sorted suffix array to allow the longest match length to be found in O(1) with O(N) precomputation.

## Example Usage

```
# Load in example tweets dataframe
import pandas as pd
example_tweet_data = pd.read_csv('example_data/example_tweet_data.csv')

from CrossEntropy import pairwise_information_flow

# Calculate information flow between users based on temporal text usage 
pairwise_information_flow(example_tweet_data, text_col = 'tweet', label_col = 'username', time_col = 'created_at')
```

## Requirements

- Python 3.x with packages:
	- numba
	- nltk (for tokenization)
	- numpy
	- LCSFinder

## Dependency on LCSFinder

The package `LCSFinder` uses a C++ backend. If this is causing issues on your machine, you can install this package without dependencies.

```
pip install --no-dependencies ProcessEntropy
```

However, you will need to ensure that the dependences `numba`, `nltk` and `numpy` are included. 

The functions which do not depend on `LCSFinder` can be accessed using the `*PythonOnly` modules. 

For example:


```
# Load in example tweets dataframe
import pandas as pd
example_tweet_data = pd.read_csv('example_data/example_tweet_data.csv')

from CrossEntropyPythonOnly import pairwise_information_flow

# Calculate information flow between users based on temporal text usage 
pairwise_information_flow(example_tweet_data, text_col = 'tweet', label_col = 'username', time_col = 'created_at')
```

**Note:** the PythonOnly variants do not perform identically, and will not pass all of the test cases. This is due to slight differences where empty source/target arrays can contribute non-zero lambda values. This behaviour was removed with the `LCSFinder` functionality.

## Installation

```
pip install ProcessEntropy
```
