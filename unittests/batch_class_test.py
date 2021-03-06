import unittest
import sys
sys.path.insert(0, '../ml_toolbox')
import gradient_descent
import numpy as np
from scipy.optimize import minimize

class BatchTest(unittest.TestCase):
	eps = 0.00001
	def test_minimize_singlevar(self):
		print("\ntest_minimize_singlevar running")
		gd = gradient_descent.Batch((lambda x: (x+1)**2-3), (lambda x: 2*(x+1)), 0.1, self.eps/float(10), 100000)
		res_batch = gd.minimize(100.0)
		print(res_batch[0])
		res_scipy = minimize((lambda x: (x+1)**2-3), 100.0, jac=(lambda x: 2*(x+1)))
		print(res_scipy)
		self.assertTrue(np.abs(res_batch[2][-1]-res_scipy.fun) < self.eps)
		self.assertTrue(np.abs(res_batch[0]-res_scipy.x[0]) < self.eps)
	def test_minimize_array(self):
		print("\ntest_minimize_array running")
		def f(x):
			return (x[0]+2)**2+(x[1]-2)**2+10
		def grad_f(x):
			dx1 = 2*(x[0]+2)
			dx2 = 2*(x[1]-2)
			return np.array([dx1,dx2])
		gd = gradient_descent.Batch(f, grad_f, 0.1, self.eps/float(10), 100)
		res_batch = gd.minimize(np.array([100.0,-10.0]))
		print(res_batch[0])
		res_scipy = minimize(f, np.array([100.0,-10.0]), jac=grad_f)
		print(res_scipy)
		self.assertTrue(np.abs(res_batch[2][-1]-res_scipy.fun) < self.eps)
		self.assertTrue((np.abs(res_batch[0]-res_scipy.x) < self.eps).all())
	def test_minimize_weirdfunc(self):
		def f(x):
			#(1-x)^2+100*(x-y)^2
			return (1-x[0])**2+100*((x[0]-x[1])**2)
		def grad_f(x):
			dx1 = 2*(1-x[0])+200*(x[0]-x[1])
			dx2 = -200*(x[0]-x[1])
			return np.array([dx1,dx2])
		gd = gradient_descent.Batch(f, grad_f, 0.1, self.eps/float(10), 1000)
		res_batch = gd.minimize(np.array([0.0,0.0]))
		print(res_batch[0])
		res_scipy = minimize(f, np.array([10.0,10.0]))
		print(res_scipy)
		self.assertTrue(np.abs(res_batch[2][-1]-res_scipy.fun) < self.eps)
		self.assertTrue((np.abs(res_batch[0]-res_scipy.x) < self.eps).all())

		
if __name__ == "__main__": 
    unittest.main()
    