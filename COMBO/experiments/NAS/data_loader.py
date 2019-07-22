import os
import numpy as np

import torch
import torch.cuda
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from COMBO.config import data_directory

from COMBO.experiments.NAS.data_config import CIFAR10_NORM_MEAN, CIFAR10_NORM_STD
from COMBO.experiments.NAS.data_config import FashionMNIST_NORM_MEAN, FashionMNIST_NORM_STD
from COMBO.experiments.NAS.data_config import MNIST_NORM_MEAN, MNIST_NORM_STD


N_VALID = 10000
NUM_WORKERS = 2
PIN_MEMORY = False


def load_cifar10(batch_size, shuffle, random_seed=None):
	num_workers = NUM_WORKERS
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=CIFAR10_NORM_MEAN, std=CIFAR10_NORM_STD)])
	data_dir = os.path.join(data_directory(), 'CIFAR10')

	train_data = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
	test_data = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
	indices = list(range(len(train_data)))
	if shuffle:
		np.random.RandomState(random_seed).shuffle(indices)
	train_idx, valid_idx = indices[:-N_VALID], indices[-N_VALID:]
	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,
	                                           num_workers=num_workers, pin_memory=PIN_MEMORY)
	valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler,
	                                           num_workers=num_workers, pin_memory=PIN_MEMORY)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
	                                          num_workers=num_workers, pin_memory=PIN_MEMORY)

	return train_loader, valid_loader, test_loader


def load_fashionmnist(batch_size, shuffle, random_seed=None):
	num_workers = NUM_WORKERS
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=FashionMNIST_NORM_MEAN, std=FashionMNIST_NORM_STD)])
	data_dir = os.path.join(data_directory(), 'FashionMNIST')

	train_data = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
	test_data = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
	indices = list(range(len(train_data)))
	if shuffle:
		np.random.RandomState(random_seed).shuffle(indices)
	train_idx, valid_idx = indices[:-N_VALID], indices[-N_VALID:]
	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,
	                                           num_workers=num_workers, pin_memory=PIN_MEMORY)
	valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler,
	                                           num_workers=num_workers, pin_memory=PIN_MEMORY)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
	                                          num_workers=num_workers, pin_memory=PIN_MEMORY)

	return train_loader, valid_loader, test_loader


def load_mnist(batch_size, shuffle, random_seed=None):
	num_workers = NUM_WORKERS
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MNIST_NORM_MEAN, std=MNIST_NORM_STD)])
	data_dir = os.path.join(data_directory(), 'MNIST')

	train_data = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
	test_data = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
	indices = list(range(len(train_data)))
	if shuffle:
		np.random.RandomState(random_seed).shuffle(indices)
	train_idx, valid_idx = indices[:-N_VALID], indices[-N_VALID:]
	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,
	                                           num_workers=num_workers, pin_memory=PIN_MEMORY)
	valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler,
	                                           num_workers=num_workers, pin_memory=PIN_MEMORY)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
	                                          num_workers=num_workers, pin_memory=PIN_MEMORY)

	return train_loader, valid_loader, test_loader

