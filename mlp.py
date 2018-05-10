from sklearn.neural_network import MLPClassifier
import load
import numpy as np

def mlp(label, identity, image, image_unlabel):
	mlp_clf = MLPClassifier(solver="sgd",max_iter = 700, learning_rate= "adaptive")
	mlp_clf.hidden_layer_sizes = (2000,)
	mlp_clf.activation = "logistic"
	train_data, test_data, train_target, test_target = load.split_data(label, identity, image)
	mlp_clf.fit(train_data, np.array(train_target).ravel())                     
	print mlp_clf.score(test_data,test_target)