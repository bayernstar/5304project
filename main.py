from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import numpy as np
import collections
import matplotlib.pyplot as plt

num_of_rows = 2925

def load_data(path):
	data = loadmat(path)
	return data['tr_labels'], data['tr_identity'].flatten(), data['tr_images'].reshape((32 * 32, num_of_rows)).T

def knn(k, train_data, train_target, test_data, test_target):
	knn_clf = neighbors.KNeighborsClassifier(n_neighbors=k)
	print k
	knn_clf.fit(train_data, np.array(train_target).ravel())
	return knn_clf.score(test_data, test_target)

def split_data(label, identity, image):
	identity_set = list(set(identity))
	train_data = []
	test_data = []
	train_target = []
	test_target = []
	train, _ = train_test_split(identity_set, test_size=0.2)
	for i in range(num_of_rows):
		if identity[i] in train:
			train_data.append(image[i])
			train_target.append(label[i])
		else:
			test_data.append(image[i])
			test_target.append(label[i])
	return train_data, test_data, train_target, test_target

def draw_labels(label):
	c = collections.Counter(label.flatten())
	y = []
	x = range(1, 8)
	for i in x:
		y.append(c[i])
	plt.plot(x, y)
	plt.xlabel('Label')
	plt.ylabel('Count')
	plt.title('Label Distribution')
	plt.savefig("label_distribution.png")

def draw_knn(label, image):
	train_data, test_data, train_target, test_target = train_test_split(image, label)
	x = range(1,16,2)
	xi = range(len(x))
	y = []
	for k in x:
		y.append(knn(k, train_data, train_target, test_data, test_target))
	xi = range(len(x))
	plt.plot(xi, y)
	plt.xticks(xi, x)
	plt.xlabel('K')
	plt.ylabel('Accuracy')
	plt.title('knn_without_identity')
	plt.ylim([0.45, 0.6])
	plt.savefig("knn_without_identity.png")
	plt.clf()

def draw_knn_identity(label, identity, image):
	train_data, test_data, train_target, test_target = split_data(label, identity, image)
	x = range(1,16,2)
	xi = range(len(x))
	y = []
	for k in x:
		y.append(knn(k, train_data, train_target, test_data, test_target))
	xi = range(len(x))
	plt.plot(xi, y)
	plt.xticks(xi, x)
	plt.xlabel('K')
	plt.ylabel('Accuracy')
	plt.title('knn_with_identity')
	plt.ylim([0.45, 0.6])
	plt.savefig("knn_with_identity.png")
	plt.clf()

def draw_k(label, identity, image):
	x = range(1,16,2)
	y = [0] * len(x)
	xi = range(len(x))
	for i in range(10):
		train_data, test_data, train_target, test_target = split_data(label, identity, image)
		for idx, k in enumerate(x):
			y[idx] += knn(k, train_data, train_target, test_data, test_target)/10.0
	xi = range(len(x))
	plt.plot(xi, y)
	plt.xticks(xi, x)
	plt.xlabel('K')
	plt.ylabel('Accuracy')
	plt.title('knn_k')
	plt.ylim([0.45, 0.6])
	plt.savefig("knn_k.png")
	plt.clf()

if __name__ == '__main__':
	label, identity, image = load_data('labeled_images')
	#draw_labels(label)
	#draw_knn(label, image)
	#draw_knn_identity(label, identity, image)
	draw_k(label, identity, image)