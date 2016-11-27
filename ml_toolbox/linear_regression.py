import gradient_descent
import numpy as np

class LinearRegression(object):
	def __init__(self, data_table, answers, rate, max_iterations): #add tolerance parameter
		self.X = data_table
		self.y = answers
		self.theta = None
		self.learning_rate = rate
		self.max_iters = max_iterations
		self.m, self.n = data_table.shape
	@staticmethod
	def cost(X, y, theta):
		return sum((np.dot(X,theta)-y)**2)/(2.0*X.shape[0])
	@staticmethod
	def cost_grad(X, y, theta):
		return np.dot(X.T, (np.dot(X,theta)-y))/float(X.shape[0])
