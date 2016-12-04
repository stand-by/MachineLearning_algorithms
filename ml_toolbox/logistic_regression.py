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
		self.X = np.insert(self.X,0,np.ones(self.m),1)
		self.m, self.n = self.X.shape


	@staticmethod
	def hypothesis(X, theta):
		return sigmoid(np.dot(X,theta))
	@staticmethod
	def cost(X, y, theta):
		return (-np.dot(y,np.log(hypothesis(X,theta)))-np.dot((1-y),np.log(1-hypothesis(X,theta))))/float(X.shape[0])
	@staticmethod
	def cost_grad(X, y, theta):
		return np.dot(X.T, (hypothesis(X,theta)-y))/float(X.shape[0])
