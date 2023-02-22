from ProcessEntropy.CrossEntropy import *
from ProcessEntropy.SelfEntropy import *

# brute force LCS
def max_match(target,source):
    '''
    Brute force function to find the longest match between a subsequence in target,
    starting at the start of target with any continuous subsequence in source.
    '''

    if len(target) ==0 or len(source)==0:
         return -1
    l = 0
    mx = 0
    for i in range(len(source)):
        k = 0
        while (k < len(target)) and (i+k) < len(source) and target[k] == source[i+k]:
            k += 1
        mx = max(mx, k)

    return mx

def brute_lambdas(target, source):
    '''
    Brute force lambdas for each possible starting sequence in target, with a history
    in the source.

    i.e. match between target[i:] and source[:i]
    '''
    return np.array([max_match(target[j:],source[:j])+1 for j in range(1,len(target)) if j<len(source)])

# functions to check
def check_get_all_self_lambdas(cases, max_len, alpha_sz):
    # Randomly generated check for get_all_self_lambdas function
    for _ in range(cases):

        # set up objects
        source = [random.randint(0, alpha_sz-1) for _ in range(random.randint(0, max_len-1))]

        assert np.all(get_all_self_lambdas(source) == brute_lambdas(source, source))

def check_get_all_lambdas(cases, max_len, alpha_sz):
    # Randomly generated checks for get_all_lambdas function (Cross Entropy)
    for _ in range(cases):
        # set up objects
        N = random.randint(0, max_len-1)
        target = [random.randint(0, alpha_sz-1) for _ in range(N)]
        source = [random.randint(0, alpha_sz-1) for _ in range(N)]

        assert np.all(get_all_lambdas(target, source, np.array([i for i in range(N)])) == brute_lambdas(target, source))


def check_find_lambda(cases, max_len, alpha_sz):
    # Randomly generated checks for find_lambda function (Cross Entropy)
    for tc in range(cases):
        # set up objects
        target = [random.randint(0, alpha_sz-1) for _ in range(random.randint(0, max_len-1))]
        source = [random.randint(0, alpha_sz-1) for _ in range(random.randint(0, max_len-1))]

        assert find_lambda(target, source) == (max_match(target,source)+1)

# Reusing old checks
def test_self_entropy():
	test_input  = [1, 2, 3, 4, 5, 1, 2, 3, 1, 4, 5, 4, 4, 1, 2, 3]
	test_output = [1, 1, 1, 1, 4, 3, 2, 2, 3, 2, 2, 2, 4, 3, 2]
	assert (self_entropy_rate(test_input, get_lambdas=True) == test_output).all()
	print(round(self_entropy_rate(test_input),2))


def test_cross_entropy():
	# Test find_lambda_jit
    # Test find_lambda_jit
    assert find_lambda(np.array([5], dtype = int),        np.array([1,2,3,4,5,3,2,4], dtype = int) ) == 2
    assert find_lambda(np.array([5,7], dtype = int),      np.array([1,2,3,4,5,3,2,4], dtype = int) ) == 2
    assert find_lambda(np.array([5,3,2 ], dtype = int),   np.array([1,2,3,4,5,3,2,4], dtype = int) ) == 4
    assert find_lambda(np.array([5,3, 5,5], dtype = int), np.array([1,2,3,4,5,3,2,4], dtype = int) ) == 3
    assert find_lambda(np.array([1,2,3], dtype = int),    np.array([1], dtype = int)) == 2
    assert find_lambda(np.array([1,2,3], dtype = int),    np.array([], dtype = int) ) == 0

	# Test get_all_lambdas cross entropy
    A = np.array([2,1,5,7,8,9,7,8,9,1,2,3,10,10,10], dtype = int)
    B = np.array([1,2,3,4,5,6,7,8,9,1,2,3,4,3,2,1,5], dtype = int)
    indices = np.array([0,0,3,3,9,9,12,12,12,12,12,14,14,14,14], dtype = int)
    L = get_all_lambdas(A,B, indices)
    assert np.mean(L == np.array([1., 1., 3., 2., 7., 6., 5., 4., 3., 2., 1., 1., 1.])) == 1
    assert np.sum(L) == 37


    # Test main function timeseries_cross_entropy
    synth_data_A = [(0,[1,2,3]),(3,[1,2,3,4,5,6,7,6,5,6,5,6]), (5,[4,5,4,5,1,2,3,10,10])]
    synth_data_B = [(2,[1,2,3,4,10,5,4,5,4]), (4,[10,10,6])]
    assert np.all(timeseries_cross_entropy(synth_data_A, synth_data_B, get_lambdas=True, please_sanitize= False) == 
            np.array([5., 4., 3., 3., 2., 1., 1., 1., 2., 1., 2., 1., 4., 4.,
            3., 2., 4., 3., 2., 3., 2.]))

def test_conditional_entropy():
	target = np.random.randint(100, size = 1000)
	source = np.random.randint(100, size = 1000)
	conditional_entropy(target, source)

def __main__():
	cases = 5000 # how many test cases for each random test
	max_len = 100 # maximum length of sequences - these are drawn uniformly at random
	alpha_sz = 5 # alphabet size

	check_get_all_self_lambdas(cases, max_len, alpha_sz)
	print('SelfEntropy working with LCSFinder.')

	check_get_all_lambdas(cases, max_len, alpha_sz)
	print('Get lambdas (cross-entropy) working with LCSFinder.')

	check_find_lambda(cases, max_len, alpha_sz)
	print('Get lambda (cross-entropy) working with LCSFinder.')

	test_self_entropy()
	print("SelfEntropy working. (Should be 1.94)")

	test_cross_entropy()
	print("CrossEntropy working.")

	test_conditional_entropy()
	print("Conditional Entropy didn't crash")

	print("""I don't have test cases from predictability yet. 
			But really, sometimes you have to live a little dangerously.""")

__main__()