import numpy as np

class BasicRegression(object):
    """
    BasicRegression class provides all necessary methods and interfaces for any regression class
    """
    def __init__(self, data_table, answers):
    	self.X = data_table
    	self.y = answers
    	self.theta = None
    	self.minimization_trace = None
    	self.m, self.n = self.X.shape

    def add_intercept(self):
		self.X = np.insert(self.X,0,np.ones(self.m),1)
		self.n += 1

    def predict(self, x0):
        raise NotImplementedError('subclasses have to override predict(self, x0)!')

	def cost(self, X, y, theta):
        raise NotImplementedError('subclasses have to override cost(self, X, y, theta)!')

	def cost_grad(self, X, y, theta):
        raise NotImplementedError('subclasses have to override cost_grad(self, X, y, theta)!')
