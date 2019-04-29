import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

from GraphDecompositionBO.experiments.NAS_binary.nas_binary_cifar10 import load_cifar10
from GraphDecompositionBO.experiments.NAS_binary.config_cifar10 import NORM_MEAN, NORM_STD, CIFAR10_CLASSES


def imshow(img, title_str):
	img = img * torch.FloatTensor(NORM_STD).view(3, 1, 1) + torch.FloatTensor(NORM_MEAN).view(3, 1, 1)  # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.title(title_str)
	plt.show()


def data_loader_check():
	# Check whether random seed properly fixes train/valid split
	batch_size = 32
	train_loader, valid_loader, test_loader = load_cifar10(batch_size=batch_size, shuffle=True, random_seed=1)
	train_label_count = list([0 for _ in range(10)])
	valid_label_count = list([0 for _ in range(10)])
	test_label_count = list([0 for _ in range(10)])
	for train_batch in train_loader:
		for train_label in train_batch[1]:
			train_label_count[train_label] += 1
	for valid_batch in valid_loader:
		for valid_label in valid_batch[1]:
			valid_label_count[valid_label] += 1
	for test_batch in test_loader:
		for test_label in test_batch[1]:
			test_label_count[test_label] += 1

	print(','.join(['%5d' % elm for elm in train_label_count]))
	print(','.join(['%5d' % elm for elm in valid_label_count]))
	print(','.join(['%5d' % elm for elm in test_label_count]))

	batch_size = 4
	train_loader, valid_loader, test_loader = load_cifar10(batch_size=batch_size, shuffle=True, random_seed=1)
	train_data_iter = iter(train_loader)
	valid_data_iter = iter(valid_loader)
	test_data_iter = iter(test_loader)
	for _ in range(2):
		train_imgs, train_labels = train_data_iter.next()
		valid_imgs, valid_labels = valid_data_iter.next()
		test_imgs, test_labels = test_data_iter.next()
		train_title_str = 'Train :' + ' '.join('%5s' % CIFAR10_CLASSES[train_labels[j]] for j in range(batch_size))
		valid_title_str = 'Valid :' + ' '.join('%5s' % CIFAR10_CLASSES[valid_labels[j]] for j in range(batch_size))
		test_title_str = 'Test :' + ' '.join('%5s' % CIFAR10_CLASSES[test_labels[j]] for j in range(batch_size))
		imshow(torchvision.utils.make_grid(train_imgs), title_str=train_title_str)
		imshow(torchvision.utils.make_grid(valid_imgs), title_str=valid_title_str)
		imshow(torchvision.utils.make_grid(test_imgs), title_str=test_title_str)
		print(train_title_str)
		print(valid_title_str)
		print(test_title_str)
