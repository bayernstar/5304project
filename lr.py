import load
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def lr(c, train_data, train_target, test_data, test_target):
	lr_clf = LogisticRegression(C=c)
	lr_clf.fit(train_data, np.array(train_target).ravel())
	return lr_clf.score(test_data, test_target)

def draw_lr(label, identity, image):
	train_data, test_data, train_target, test_target = load.split_data(label, identity, image)
	x = [0.01, 0.1, 1]
	xi = range(len(x))
	y = []
	for c in x:
		y.append(lr(c, train_data, train_target, test_data, test_target))
	plt.plot(xi, y)
	plt.xticks(xi, x)
	plt.xlabel('C')
	plt.ylabel('Accuracy')
	plt.title('logistic_regression')
	plt.ylim([0.7, 0.75])
	plt.savefig("logistic_regression.png")
	plt.clf()

def get_lr(label, identity, image):
	train_data, test_data, train_target, test_target = load.split_data(label, identity, image)
	lr_clf = LogisticRegression(C=0.01)
	lr_clf.fit(train_data, np.array(train_target).ravel())
	return lr_clf

def draw_lr_identity(label, image):
	train_data, test_data, train_target, test_target = train_test_split(image, label, test_size=0.2)
	x = [0.01, 0.1, 1]
	xi = range(len(x))
	y = []
	for c in x:
		y.append(lr(c, train_data, train_target, test_data, test_target))
	plt.plot(xi, y)
	plt.xticks(xi, x)
	plt.xlabel('C')
	plt.ylabel('Accuracy')
	plt.title('logistic_regression')
	#plt.ylim([0.7, 0.75])
	plt.savefig("logistic_regression.png")
	plt.clf()

def visualize(label, identity, image):
	train_data, test_data, train_target, test_target = load.split_data(label, identity, image)
	x = range(1,8)
	xi = range(len(x))
	lr_clf = LogisticRegression(C=0.01)
	lr_clf.fit(train_data, np.array(train_target).ravel())
	lr_prediction = lr_clf.predict(test_data)
	y1 = [0] * 7
	y2 = [0] * 7
	for i in range(len(test_target)):
		y1[test_target[i][0]-1] += 1
		if lr_prediction[i] == test_target[i][0]:
			y2[test_target[i][0]-1] += 1
	y = []
	for i in range(7):
		y.append(float(y2[i])/y1[i])
	plt.plot(xi, y)
	plt.xticks(xi, x)
	plt.xlabel('label')
	plt.ylabel('accuracy')
	plt.title('labels')
	plt.savefig("labels.png")
	plt.clf()