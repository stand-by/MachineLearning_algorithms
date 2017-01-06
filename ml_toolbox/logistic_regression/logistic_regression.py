from basic_regression import BasicRegression
import gradient_descent
import numpy as np
import functools
from scipy import optimize

def sigmoid(x):
	return 1.0/(1+np.exp(-x))

class LogisticRegression(BasicRegression):
	"""
	LogisticRegression class provides model for solving binary classification problem based on labeled data.
	It has to be constructed with X matrix and y vector (class adds bias column automaticly).
	LogisticRegression has few useful methods: train_batch - fits appropriate model's parameters using gradient descent
	train_nelder_mead - calculates parameters using Nelder-Mead minimization method
	predict - returns label(0/1) on given vector of parameters
	probability - returns the probability of input vector belongs to positive class
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
		batch = gradient_descent.Batch(J, J_grad, rate, tolerance, max_iters)
		self.minimization_trace = batch.minimize(inital_guess)
		self.theta = self.minimization_trace[0]

	def train_nelder_mead(self, max_iters, inital_guess):
		"""
		max_iters=maximum number of iterations in gradient descent
		inital_guess=point for minimazation to start with
		"""
		J = functools.partial(self.cost,self.X,self.y)
		J_grad = functools.partial(self.cost_grad,self.X,self.y)
		self.theta = optimize.fmin(J, x0=inital_guess, maxiter=max_iters, full_output=False, disp=False)

	def probability(self, x0):
		"""
		returns the probability of given vector x0 belongs to positive class
		"""
		x0 = np.insert(x0,0,1.0)
		return self.hypothesis(x0,self.theta)

	def predict(self, x0):
		"""
		returns model's prediction for given vector x0 as argument
		"""
		klass = 1 if self.probability(x0) >= 0.5 else 0
		return klass

	def hypothesis(self, X, theta):
		return sigmoid(np.dot(X,theta))

	def cost(self, X, y, theta):
		return (np.dot(-y.T,np.log(sigmoid(np.dot(X,theta))))-np.dot((1-y).T,np.log(1-sigmoid(np.dot(X,theta)))))/float(X.shape[0])

	def cost_grad(self, X, y, theta):
		return np.dot(X.T, (sigmoid(np.dot(X,theta))-y))/float(X.shape[0])
