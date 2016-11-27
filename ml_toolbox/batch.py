import numpy as np

class Batch(object):
	def __init__(self, func, grad, rate, num_iter):
		self.function = func
		self.gradient = grad
		self.iterations = num_iter
		self.learning_rate = rate


if __name__ == '__main__':
	