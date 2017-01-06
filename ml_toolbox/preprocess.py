import numpy as np

def get_normalized_and_scaled(X):
	return (X - np.mean(X,axis=0))/np.std(X,axis=0)
def get_normalized(X):
	return (X - np.mean(X,axis=0))
def get_scaled(X):
	return X/np.std(X,axis=0)

def split_dataset(X, ratio):
	train_size = int(len(X)*ratio)
	train_set = []
	copy = list(X)
	while len(train_set) < train_size:
		index = np.random.randint(0,len(copy))
		train_set.append(copy.pop(index))
	return (np.array(train_set), np.array(copy))
def split_dataset_train_cv_test(X):
	train, cv_test = split_dataset(X,0.6)
	cv, test = split_dataset(cv_test,0.5)
	return (train, cv, test)

def readfile(filename, separator=','):
	dataset = []
	f = open(filename)
	for line in f:
		row = line.rstrip().split(separator)
		if row==['']: continue
		lst = [float(word) for word in row]
		dataset.append(lst)
	return dataset
def readfile_headed(filename, header_separator=',', body_separator=","):
	labels = []
	dataset = []
	f = open(filename)
	labels = f.readline().rstrip().split(header_separator)
	for line in f:
		row = line.rstrip().split(body_separator)
		if row==['']: continue
		lst = [float(word) for word in row]
		dataset.append(lst)
	return (labels, dataset)

def polynomial_combiantion(x1col, x2col, degrees):
	prohibited = np.hstack((np.array([x1col]).T,np.array([x2col]).T))
	prohibited = map_to_degrees(prohibited,degrees)
	out = np.ones((x1col.shape[0], 1))
	for i in range(1, degrees+1):
		for j in range(0, i+1):
			term1 = x1col ** (i-j)
			term2 = x2col ** (j)
			term = (term1 * term2).reshape(term1.shape[0], 1)
			if term.T.tolist()[0] in prohibited.T.tolist(): continue
			out = np.hstack((out, term))
	return out[:,1:]
def map_to_degrees(X, k):
	X_copy = np.copy(X)
	for i in range(2,k+1):
		tpl = (X_copy, np.power(X,i))
		X_copy = np.hstack(tpl)
	return X_copy
def map_to_polynomial(X, k):
	X_copy = np.copy(X)
	n = X.shape[1]
	for i in range(n):
		for j in range(i+1,n):
			mapped_term = polynomial_combiantion(X[:,i], X[:,j], k)
			X_copy = np.hstack((X_copy,mapped_term))
	X_copy = np.hstack((X_copy[:,n:], map_to_degrees(X,k)))
	return X_copy
