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
