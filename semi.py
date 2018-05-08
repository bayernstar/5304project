import load
from sklearn.semi_supervised import LabelPropagation
import numpy as np
import matplotlib.pyplot as plt

def lp(train_data, train_target, test_data, test_target):
	lp_clf = LabelPropagation()
	lp_clf.fit(train_data, train_target)
	return lp_clf.score(test_data, test_target)

def draw_lp(label, identity, image, image_unlabel):
	images = np.vstack((image, image_unlabel))
	padding = np.ones((image_unlabel.shape[0],))
	labels = np.concatenate((label.ravel(), padding), axis = 0)
	train_data, test_data, train_target, test_target = load.split_data(label, identity, image)
	print lp(images, labels, test_data, test_target)