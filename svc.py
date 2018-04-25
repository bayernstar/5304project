import load
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV

def svc_linear(c, train_data, train_target, test_data, test_target):
	ovr_clf = OneVsRestClassifier(LinearSVC(C=c))
	ovr_clf.fit(train_data, np.array(train_target).ravel())
	return ovr_clf.score(test_data, test_target)

def draw_svc_linear(label, identity, image):
	train_data, test_data, train_target, test_target = load.split_data(label, identity, image)
	x = [0.1, 1, 10]
	xi = range(len(x))
	y = []
	for c in x:
		y.append(svc_linear(c, train_data, train_target, test_data, test_target))
	plt.plot(xi, y)
	plt.xticks(xi, x)
	plt.xlabel('C')
	plt.ylabel('Accuracy')
	plt.title('svc_linear')
	plt.ylim([0.70, 0.75])
	plt.savefig("svc_linear.png")
	plt.clf()

def svc_rbf(train_data, train_target, test_data, test_target):
	rbf_clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), {'C': [100], 'gamma': [0.01] })
	rbf_clf.fit(train_data, np.array(train_target).ravel())
	return rbf_clf.score(test_data, test_target)

def draw_svc_rbf_label(label, identity, image):
	x = range(1,11)
	y = []
	for c in x:
		train_data, test_data, train_target, test_target = load.split_data(label, identity, image)
		train_data,test_data = load.pca_data(train_data, train_data, test_data)
		y.append(svc_rbf(train_data, train_target, test_data, test_target))
	plt.plot(x, y)
	plt.xlabel('Run')
	plt.ylabel('Accuracy')
	plt.title('svc_rbf_label')
	plt.ylim([0.70, 0.80])
	plt.savefig("svc_rbf_label.png")
	plt.clf()

def draw_svc_rbf(label, identity, image, image_unlabel):
	x = range(1,11)
	y = []
	for c in x:
		train_data, test_data, train_target, test_target = load.split_data(label, identity, image)
		train_data,test_data = load.pca_data(np.vstack((train_data, image_unlabel)), train_data, test_data)
		y.append(svc_rbf(train_data, train_target, test_data, test_target))
	plt.plot(x, y)
	plt.xlabel('Run')
	plt.ylabel('Accuracy')
	plt.title('svc_rbf')
	plt.ylim([0.70, 0.80])
	plt.savefig("svc_rbf.png")
	plt.clf()