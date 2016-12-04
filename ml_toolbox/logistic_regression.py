import gradient_descent
import numpy as np

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
		#temporary use of scipy
		from scipy.optimize import minimize
		res_scipy = minimize(J, inital_guess, jac=J_grad)
		self.theta = res_scipy.x
	@staticmethod
	def hypothesis(X, theta):
		return sigmoid(np.dot(X,theta))
	@staticmethod
	def cost(X, y, theta):
		return (np.dot(-y.T,np.log(LogisticRegression.hypothesis(X,theta)))-np.dot((1-y).T,np.log(1-LogisticRegression.hypothesis(X,theta))))/float(X.shape[0])
	@staticmethod
	def cost_grad(X, y, theta):
		return np.dot(X.T, (LogisticRegression.hypothesis(X,theta)-y))/float(X.shape[0])
