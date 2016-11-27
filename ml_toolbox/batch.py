import numpy as np
from scipy.optimize import minimize

class Batch(object):
	def __init__(self, func, grad, rate, num_iter):
		self.function = func
		self.gradient = grad
		self.iterations = num_iter
		self.learning_rate = rate
	def minimize(self, initial_guess):
		x = initial_guess
		for i in range(self.iterations):
			x = x - self.learning_rate*self.gradient(x)
		return {'x_max': x, 'y_max': self.function(x)}


if __name__ == '__main__':
	gd = Batch((lambda x: (x+1)**2-3), (lambda x: 2*(x+1)), 0.1, 100)
	print(gd.minimize(100.0))
	print(minimize((lambda x: (x+1)**2-3), 100.0, jac=(lambda x: 2*(x+1))))
	print("")

	def f(x):
		return (x[0]+2)**2+(x[1]-2)**2+10
	def grad_f(x):
		dx1 = 2*(x[0]+2)
		dx2 = 2*(x[1]-2)
		return np.array([dx1,dx2])
	gd = Batch(f, grad_f, 0.1, 100)
	print(gd.minimize(np.array([100.0,-10.0])))
	print(minimize(f, np.array([100.0,-10.0]), jac=grad_f))
	print("")