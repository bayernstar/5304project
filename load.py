from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

num_of_rows = 2925
num_of_rows_unlabel = 98058

def load_data(path, label=True, identity=True):
	data = loadmat(path)
	if label:
		if identity:
			return data['tr_labels'], data['tr_identity'].flatten(), data['tr_images'].reshape((32 * 32, num_of_rows)).T
		else:
			return data['tr_labels'], data['tr_images'].reshape((32 * 32, 72799)).T
	else:
		return data['unlabeled_images'].reshape((32 * 32, num_of_rows_unlabel )).T

def split_data(label, identity, image):
	identity_set = list(set(identity))
	train_data = []
	test_data = []
	train_target = []
	test_target = []
	train, _ = train_test_split(identity_set, test_size=0.1)
	for i in range(num_of_rows):
		if identity[i] in train:
			train_data.append(image[i])
			train_target.append(label[i])
		else:
			test_data.append(image[i])
			test_target.append(label[i])
	return train_data, test_data, train_target, test_target

def pca_data(train_sum, train, test):
    pca_clf = PCA(150, whiten=True)
    pca_clf.fit(train_sum)
    return pca_clf.transform(train), pca_clf.transform(test)