import numpy as np
from scipy.optimize import minimize

class Batch(object):
	def __init__(self, func, grad, rate, num_iter):
		self.function = func
		self.gradient = grad
		self.iterations = num_iter
		self.learning_rate = rate
	def minimize(self, initial_guess):
		x = initial_guess
		for i in range(self.iterations):
			x = x - self.learning_rate*self.gradient(x)
		return {'x_max': x, 'y_max': self.function(x)}


if __name__ == '__main__':
	