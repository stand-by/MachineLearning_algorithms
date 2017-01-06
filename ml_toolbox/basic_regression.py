import numpy as np

class BasicRegression(object):
    def __init__(self, data_table, answers):
    	self.X = data_table
    	self.y = answers
    	self.theta = None
    	self.minimization_trace = None
    	self.m, self.n = self.X.shape

    def add_intercept(self):
		self.X = np.insert(self.X,0,np.ones(self.m),1)
		self.n += 1
