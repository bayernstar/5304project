import load
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def svc_rbf(train_data, train_target, test_data, test_target):
	rbf_clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), {'C': [100], 'gamma': [0.01] })
	rbf_clf.fit(train_data, np.array(train_target).ravel())
	return rbf_clf

def lr(train_data, train_target, test_data, test_target):
	lr_clf = LogisticRegression(C=0.01)
	lr_clf.fit(train_data, np.array(train_target).ravel())
	return lr_clf

def svc_linear(train_data, train_target, test_data, test_target):
	ovr_clf = OneVsRestClassifier(LinearSVC(C=1))
	ovr_clf.fit(train_data, np.array(train_target).ravel())
	return ovr_clf

def mlp(train_data, train_target, test_data, test_target):
	mlp_clf = MLPClassifier(solver="sgd",max_iter = 700, learning_rate= "adaptive")
	mlp_clf.hidden_layer_sizes = (2000,)
	mlp_clf.activation = "logistic"
	mlp_clf.fit(train_data, np.array(train_target).ravel())                     
	return mlp_clf

def draw_ensemble(label, identity, image, image_unlabel):
	train_data, test_data, train_target, test_target = load.split_data(label, identity, image)
	mlp_clf = mlp(train_data, train_target, test_data, test_target)
	mlp_pred = mlp_clf.predict(test_data)
	lr_clf = lr(train_data, train_target, test_data, test_target)
	lr_pred = lr_clf.predict(test_data)
	train_data,test_data = load.pca_data(np.vstack((train_data, image_unlabel)), train_data, test_data)
	rbf_clf = svc_rbf(train_data, train_target, test_data, test_target)
	rbf_pred = rbf_clf.predict(test_data)
	result = []
	c1 = 0
	for i in range(len(lr_pred)):
		if lr_pred[i] == mlp_pred[i] or mlp_pred[i] == rbf_pred[i]:
			result.append(mlp_pred[i])
		else:
			result.append(rbf_pred[i])
		if result[i] == test_target[i]:
			c1 += 1
	print "accuracy is: " + str(c1/float(len(lr_pred)))