from basic_regression import BasicRegression
import gradient_descent
import numpy as np
import functools
from scipy import optimize

def sigmoid(x):
	return 1.0/(1+np.exp(-x))

class LogisticRegression(BasicRegression):
	def __init__(self, data_table, answers):
		BasicRegression.__init__(self,data_table,answers)
		self.add_intercept()

	def train_batch(self, rate, tolerance, max_iters, inital_guess):
		J = functools.partial(self.cost,self.X,self.y)
		J_grad = functools.partial(self.cost_grad,self.X,self.y)
		batch = gradient_descent.Batch(J, J_grad, rate, tolerance, max_iters)
		self.minimization_trace = batch.minimize(inital_guess)
		self.theta = self.minimization_trace[0]

	def train_nelder_mead(self, max_iters, inital_guess):
		J = functools.partial(self.cost,self.X,self.y)
		J_grad = functools.partial(self.cost_grad,self.X,self.y)
		self.theta = optimize.fmin(J, x0=inital_guess, maxiter=max_iters, full_output=False, disp=False)

	def probability(self, x0):
		x0 = np.insert(x0,0,1.0)
		return self.hypothesis(x0,self.theta)

	def predict(self, x0):
		klass = 1 if self.probability(x0) >= 0.5 else 0
		return klass

	def hypothesis(self, X, theta):
		return sigmoid(np.dot(X,theta))

	def cost(self, X, y, theta):
		return (np.dot(-y.T,np.log(sigmoid(np.dot(X,theta))))-np.dot((1-y).T,np.log(1-sigmoid(np.dot(X,theta)))))/float(X.shape[0])

	def cost_grad(self, X, y, theta):
		return np.dot(X.T, (sigmoid(np.dot(X,theta))-y))/float(X.shape[0])
