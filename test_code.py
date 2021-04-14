# This is a temporary testing file until I get around to adding an actual testing package.


from ProcessEntropy.SelfEntropy import *
from ProcessEntropy.CrossEntropy import *


def test_self_entropy():
	test_input  = [1, 2, 3, 4, 5, 1, 2, 3, 3, 4, 5, 4, 4, 1, 2, 3]
	test_output = [0, 1, 1, 1, 1, 4, 3, 2, 4, 3, 2, 2, 2, 4, 3, 2] 
	assert (self_entropy_rate(test_input, get_lambdas=True) == test_output).all()
	print(self_entropy_rate(test_input))


def test_cross_entropy():
	# Test find_lambda_jit
	assert find_lambda_jit(np.array([5], dtype = int),        np.array([1,2,3,4,5,3,2,4], dtype = int) ) == 2
	assert find_lambda_jit(np.array([5,7], dtype = int),      np.array([1,2,3,4,5,3,2,4], dtype = int) ) == 2
	assert find_lambda_jit(np.array([5,3,2 ], dtype = int),   np.array([1,2,3,4,5,3,2,4], dtype = int) ) == 3
	assert find_lambda_jit(np.array([5,3, 5,5], dtype = int), np.array([1,2,3,4,5,3,2,4], dtype = int) ) == 3
	assert find_lambda_jit(np.array([1,2,3], dtype = int),    np.array([1], dtype = int)) == 2
	assert find_lambda_jit(np.array([1,2,3], dtype = int),    np.array([], dtype = int) ) == 1

	# Test get_all_lambdas cross entropy
	A = np.array([2,1,5,7,8,9,7,8,9,1,2,3,10,10,10], dtype = int)
	B = np.array([1,2,3,4,5,6,7,8,9,1,2,3,4,3,2,1,5], dtype = int)
	indices = np.array([0,0,3,3,9,9,12,12,12,12,12,14,14,14,14], dtype = int)
	L = get_all_lambdas(A,B, indices, np.zeros(len(A)).astype(int))
	assert np.mean(L == np.array([1., 1., 1., 1., 3., 2., 7., 6., 5., 4., 3., 2., 1., 1., 1.])) == 1
	assert np.sum(L) == 39


	# Test main function timeseries_cross_entropy
	synth_data_A = [(0,[1,2,3]),(3,[1,2,3,4,5,6,7,6,5,6,5,6]), (5,[4,5,4,5,1,2,3,10,10])]
	synth_data_B = [(2,[1,2,3,4,10,5,4,5,4]), (4,[10,10,6])]
	assert np.mean(timeseries_cross_entropy(synth_data_A, synth_data_B, get_lambdas=True, please_sanitize= False) == 
	        np.array([1., 1., 1., 5., 4., 3., 3., 2., 1., 1., 1., 2., 1., 2., 1., 4., 4.,
	       3., 2., 4., 3., 2., 2., 2.])) == 1

def test_conditional_entropy():
	target = np.random.randint(100, size = 1000)
	source = np.random.randint(100, size = 1000)
	conditional_entropy(target, source)

def __main__():
	test_self_entropy()
	print("SelfEntropy working. (Should be 1.83)")

	test_cross_entropy()
	print("CrossEntropy working.")

	test_conditional_entropy()
	print("Conditional Entropy didn't crash")

	print("""I don't have test cases from predictability yet. 
			But really, sometimes you have to live a little dangerously.""")



__main__()