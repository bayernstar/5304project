import knn
import load
import lr
import svc
import semi
import ensemble
import mlp

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
	#semi.draw_lp(label, identity, image, image_unlabel)
	#semi.draw_nb(label, identity, image, image_unlabel)
	#mlp.mlp(label, identity, image, image_unlabel)
	ensemble.draw_ensemble(label, identity, image, image_unlabel)