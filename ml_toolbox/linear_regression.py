import numpy as np

class LinearRegression(object):
	def __init__(self, data_table, answers, rate, max_iterations): #add tolerance parameter
		self.X = data_table
		self.y = answers
		self.learning_rate = rate
		self.max_iters = max_iterations
		self.m, self.n = data_table.shape
	@staticmethod
	def cost(X, y, theta):
		return sum((np.dot(X,theta)-y)**2)/(2.0*X.shape[0])
