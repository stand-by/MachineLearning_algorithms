from ml_toolbox.basic_regression import BasicRegression
from ml_toolbox import gradient_descent
import functools
import numpy as np

class LinearRegression(BasicRegression):
	"""
	LinearRegression class provides model for building predictions based on design matrix X and given answers y.
	It has to be constructed with X matrix and y vector (class adds bias column automaticly).
	LinearRegression has three useful methods: train_batch - fits appropriate parameters using gradient descent
	train_normal_equation - calculates parameters using equation
	predict - returns prediction on given vector of parameters
	"""
	def __init__(self, data_table, answers):
		"""
		data_table=design matrix, without column of ones
		answers=labels for data to train
		"""
		BasicRegression.__init__(self,data_table,answers)
		self.add_intercept()

	def train_batch(self, rate, tolerance, max_iters, inital_guess):
		"""
		rate=learning rate which will be used in gradient descent
		tolerance=precision, also needs by gradient descent
		max_iters=maximum number of iterations in gradient descent
		inital_guess=point for minimazation to start with
		"""
		J = functools.partial(self.cost,self.X,self.y)
		J_grad = functools.partial(self.cost_grad,self.X,self.y)
		batch = gradient_descent.Batch(J,J_grad,rate,tolerance,max_iters)
		self.minimization_trace = batch.minimize(inital_guess)
		self.theta = self.minimization_trace[0]

	def train_normal_equation(self):
		"""
		just solves normal equation and finds thethas (really slow for big design matrix)
		"""
		self.theta = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(self.X), self.X)), np.transpose(self.X)), self.y)

	def predict(self, x0):
		"""
		returns model's prediction for given vector x0 as argument
		"""
		x0 = np.insert(x0,0,1.0)
		return np.dot(x0,self.theta)

	def cost(self, X, y, theta):
		return sum((np.dot(X,theta)-y)**2)/(2.0*X.shape[0])

	def cost_grad(self, X, y, theta):
		return np.dot(X.T, (np.dot(X,theta)-y))/float(X.shape[0])
