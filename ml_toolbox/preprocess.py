import numpy as np

def get_normalized_and_scaled(X):
	return (X - np.mean(X,axis=0))/np.std(X,axis=0)
def get_normalized(X):
	return (X - np.mean(X,axis=0))
def get_scaled(X):
	return X/np.std(X,axis=0)