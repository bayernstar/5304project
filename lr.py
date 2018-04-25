import load
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

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