import unittest
import sys
sys.path.insert(0, '../ml_toolbox')
import gradient_descent
import numpy as np
from scipy.optimize import minimize

class BatchTest(unittest.TestCase):
	eps = 0.00001
	def test_minimize_singlevar(self):
		gd = gradient_descent.Batch((lambda x: (x+1)**2-3), (lambda x: 2*(x+1)), 0.1, 100)
		res_batch = gd.minimize(100.0)
		print(res_batch)
		res_scipy = minimize((lambda x: (x+1)**2-3), 100.0, jac=(lambda x: 2*(x+1)))
		print(res_scipy)
		self.assertTrue(np.abs(res_batch['y_max']-res_scipy.fun) < self.eps)
		self.assertTrue(np.abs(res_batch['x_max']-res_scipy.x[0]) < self.eps)


if __name__ == "__main__": 
    unittest.main()