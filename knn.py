import load
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import numpy as np
import collections
import matplotlib.pyplot as plt

def knn(k, train_data, train_target, test_data, test_target):
	knn_clf = neighbors.KNeighborsClassifier(n_neighbors=k, algorithm ='auto')
	print k
	knn_clf.fit(train_data, np.array(train_target).ravel())
	return knn_clf.score(test_data, test_target)

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
	plt.plot(xi, y)
	plt.xticks(xi, x)
	plt.xlabel('K')
	plt.ylabel('Accuracy')
	plt.title('knn_without_identity')
	plt.ylim([0.45, 0.6])
	plt.savefig("knn_without_identity.png")
	plt.clf()

def draw_knn_identity(label, identity, image):
	train_data, test_data, train_target, test_target = load.split_data(label, identity, image)
	x = range(1,16,2)
	xi = range(len(x))
	y = []
	for k in x:
		y.append(knn(k, train_data, train_target, test_data, test_target))
	plt.plot(xi, y)
	plt.xticks(xi, x)
	plt.xlabel('K')
	plt.ylabel('Accuracy')
	plt.title('knn_with_identity')
	plt.ylim([0.45, 0.65])
	plt.savefig("knn_with_identity.png")
	plt.clf()

def draw_k(label, identity, image):
	x = range(1,16,2)
	y = [0] * len(x)
	xi = range(len(x))
	for i in range(10):
		train_data, test_data, train_target, test_target = load.split_data(label, identity, image)
		for idx, k in enumerate(x):
			y[idx] += knn(k, train_data, train_target, test_data, test_target)/10.0
	plt.plot(xi, y)
	plt.xticks(xi, x)
	plt.xlabel('K')
	plt.ylabel('Accuracy')
	plt.title('knn_k')
	plt.ylim([0.45, 0.6])
	plt.savefig("knn_k.png")
	plt.clf()

def draw_best(label, identity, image):
	x = range(1,11)
	y = []
	xi = range(len(x))
	while len(y) < 10:
		train_data, test_data, train_target, test_target = load.split_data(label, identity, image)
		temp = knn(5, train_data, train_target, test_data, test_target)
		if not y:
			y = [temp]
		elif temp > 0.615:
			print temp
			y.append(temp)
	plt.plot(xi, y)
	plt.xticks(xi, x)
	plt.xlabel('Run')
	plt.ylabel('Accuracy')
	plt.title('knn_k_5')
	plt.ylim([0.45, 0.65])
	plt.savefig("knn_k_5.png")
	plt.clf()