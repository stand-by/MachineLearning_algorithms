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
