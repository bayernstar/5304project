import knn
import load
import lr
import svc
import semi
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import neighbors
import numpy as np
from scipy.io import savemat
from sklearn.semi_supervised import LabelPropagation

if __name__ == '__main__':
	label, identity, image = load.load_data('labeled_images')
	image_unlabel = load.load_data('unlabeled_images', False)
	"""
	project1
	knn.draw_labels(label)
	knn.draw_knn(label, image)
	knn.draw_knn_identity(label, identity, image)
	knn.draw_k(label, identity, image)
	knn.draw_best(label, identity, image)
	
	project2
	lr.draw_lr(label, identity, image)
	svc.draw_svc_linear(label, identity, image)
	svc.draw_svc_rbf_label(label, identity, image)
	svc.draw_svc_rbf(label, identity, image, image_unlabel)
	"""
	#project3
	semi.draw_lp(label, identity, image, image_unlabel)