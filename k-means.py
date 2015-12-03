from sklearn.cluster import KMeans
import read_data
import numpy as np

def Kmeans(initarray, dataset, label):
	km = KMeans(n_clusters = 20, init = initarray, n_init = 1)
	prediction = km.fit_predict(dataset)
	return prediction

def cluster_mean_array(dataset, label):
	mean_array = {}
	for i in range(0, len(label)):
		if mean_array.has_key(label[i]):
			mean_array[label[i]] += dataset[i]
		else:
			mean_array[label[i]] = dataset[i]
	mean_matrix = []
	for i in range(0, 200):
		mean_matrix.append(mean_array[i])
	mean_matrix = np.array(mean_matrix)
	mean_matrix = mean_matrix/15
	return mean_matrix

def cluster(dataset, mean_matrix):
	return label

if __name__ == '__main__':
	
