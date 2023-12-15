import os
import cv2
import numpy as np 
from matplotlib import pyplot as plt
from src.dataset import classID_inc_normal, classID


def plot_confusion_matrix(confusion_matrix, num_classes, class_names, path2save, taskname):
	# check if directory exists
	dirname = os.path.dirname(path2save)
	if not os.path.exists(dirname):
		os.makedirs(dirname, exist_ok = True)

	plt.figure(figsize = (12,10))
	plt.imshow(confusion_matrix, cmap = 'Blues')
	# add numbers to confusion matrix
	for i in range(num_classes):
		for j in range(num_classes):
			plt.text(j, i, "{:.3f}".format(confusion_matrix[i,j]*100), ha = 'center', va = 'center', color = 'black', fontsize = 8)

	plt.colorbar()
	plt.xticks(np.arange(num_classes), [class_names[i] for i in range(num_classes)], rotation = 90, fontsize = 8)
	plt.yticks(np.arange(num_classes), [class_names[i] for i in range(num_classes)], fontsize = 8)
	plt.xlabel('Predicted')
	plt.ylabel('Ground Truth')
	plt.title(f"Confusion matrix for {taskname}(%)")
	plt.savefig(path2save)    
	plt.close()

	img = cv2.imread(path2save)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img

