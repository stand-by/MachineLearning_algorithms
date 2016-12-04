import gradient_descent
import numpy as np
from scipy import optimize

def sigmoid(x):
	return 1.0/(1+np.exp(-x))

class LogisticRegression(object):
	def __init__(self, data_table, answers):
		self.X = data_table
		self.y = answers
		self.theta = None
		self.minimization_trace = None
		self.X = np.insert(self.X,0,np.ones(self.X.shape[0]),1)
		self.m, self.n = self.X.shape
	def train_batch(self, rate, tolerance, max_iters, inital_guess):
		J = lambda t: LogisticRegression.cost(self.X,self.y,t)
		J_grad = lambda t: LogisticRegression.cost_grad(self.X,self.y,t)
		batch = gradient_descent.Batch(J, J_grad, rate, tolerance, max_iters)
		self.minimization_trace = batch.minimize(inital_guess)
		self.theta = self.minimization_trace[0]
	def train_nelder_mead(self, max_iters, inital_guess):
		J = lambda t: LogisticRegression.cost(self.X,self.y,t)
		J_grad = lambda t: LogisticRegression.cost_grad(self.X,self.y,t)
		self.theta = optimize.fmin(J, x0=inital_guess, maxiter=max_iters, full_output=False, disp=False)
	@staticmethod
	def hypothesis(X, theta):
		return sigmoid(np.dot(X,theta))
	@staticmethod
	def cost(X, y, theta):
		return (np.dot(-y.T,np.log(LogisticRegression.hypothesis(X,theta)))-np.dot((1-y).T,np.log(1-LogisticRegression.hypothesis(X,theta))))/float(X.shape[0])
	@staticmethod
	def cost_grad(X, y, theta):
		return np.dot(X.T, (LogisticRegression.hypothesis(X,theta)-y))/float(X.shape[0])
