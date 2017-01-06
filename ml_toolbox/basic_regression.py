class BasicRegression(object):
    def __init__(self, data_table, answers):
    	self.X = data_table
    	self.y = answers
    	self.theta = None
    	self.minimization_trace = None
    	self.m, self.n = self.X.shape
