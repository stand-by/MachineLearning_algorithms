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
	def train_batch(self, inital_guess):
		self.X = np.insert(self.X,0,np.ones(m),1)
		J = lambda t: LinearRegression.cost(self.X,self.y,t)
		J_grad = lambda t: LinearRegression.cost_grad(self.X,self.y,t)
		batch = gradient_descent.Batch(J,J_grad,self.rate,self.max_iterations)
		self.theta = batch.minimize(inital_guess)
	def predict(self, x0):
		x0 = np.insert(x0,0,1.0)
		return x0*self.theta
	@staticmethod
	def cost(X, y, theta):
		return sum((np.dot(X,theta)-y)**2)/(2.0*X.shape[0])
	@staticmethod
	def cost_grad(X, y, theta):
		return np.dot(X.T, (np.dot(X,theta)-y))/float(X.shape[0])
