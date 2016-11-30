import gradient_descent
import numpy as np

class LinearRegression(object):
	def __init__(self, data_table, answers):
		self.X = data_table
		self.y = answers
		self.theta = None
		self.minimization_trace = None
		self.m, self.n = self.X.shape
		self.X = np.insert(self.X,0,np.ones(self.m),1)
	def train_batch(self, rate, tolerance, max_iters, inital_guess):
		J = lambda t: LinearRegression.cost(self.X,self.y,t)
		J_grad = lambda t: LinearRegression.cost_grad(self.X,self.y,t)
		batch = gradient_descent.Batch(J,J_grad,rate,tolerance,max_iters)
		self.minimization_trace = batch.minimize(inital_guess)
		self.theta = self.minimization_trace[0]
	def train_normal_equation(self):
		self.theta = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(self.X), self.X)), np.transpose(self.X)), self.y)
	def predict(self, x0):
		x0 = np.insert(x0,0,1.0)
		return np.dot(x0,self.theta)
	@staticmethod
	def cost(X, y, theta):
		return sum((np.dot(X,theta)-y)**2)/(2.0*X.shape[0])
	@staticmethod
	def cost_grad(X, y, theta):
		return np.dot(X.T, (np.dot(X,theta)-y))/float(X.shape[0])
