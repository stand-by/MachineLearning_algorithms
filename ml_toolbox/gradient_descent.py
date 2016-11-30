class Batch(object):
	"""
	Batch class provides simple interface to gradient descent which is able to find local optimum.
	The class has a constructor that takes user's function as callable which takes np.array as argument
	and callable gradient of this function that returns gradient as np.array in certain point;
	also, constructor takes learning rate and maximum number of iterations to converge.
	Minimize method simply takes initial point to start descent and returns local optimum.
	"""
	def __init__(self, func, grad, rate, tolerance, max_iters):
		"""
		func=your callable function that takes np.array and returns np.array;
		grad=callable gradient to your function that takes np.array and returns np.array
		rate=model parameter that defines learning rate
		num_iter=maximum number of iterations
		"""
		self.function = func
		self.gradient = grad
		self.max_iterations = max_iters
		self.learning_rate = rate
		self.tolerance = tolerance
	def minimize(self, initial_guess):
		x = initial_guess
		for i in range(self.iterations):
			x = x - self.learning_rate*self.gradient(x)
		return {'x_max': x, 'y_max': self.function(x)}
		