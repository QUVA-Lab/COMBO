import time
import sys
import argparse
from pthflops import count_ops
import numpy as np

import torch
import torch.optim as optim
import torch.cuda


from GraphDecompositionBO.experiments.NAS_binary.generate_architecture import valid_net_topo, NASBinaryCNN
from GraphDecompositionBO.experiments.NAS_binary.data_loader import load_cifar10, load_fashionmnist, load_mnist


N_COMPARE = 10


def train(model, n_epochs, train_loader, eval_loader, device, display=False):
	cuda = torch.cuda.is_available() and device is not None

	if cuda:
		model.cuda(device=device)

	eval_acc_list = []

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), weight_decay=5e-5)

	for epoch in range(n_epochs):
		running_loss = 0.0
		if display:
			sys.stdout.write(
				time.strftime('Train : %H:%M:%S', time.gmtime()) + (' %3d ~ ' % len(train_loader)) + (' ' * 12))
		for i, train_data in enumerate(train_loader):
			train_inputs, train_labels = train_data
			if cuda:
				train_inputs = train_inputs.cuda(device=device)
				train_labels = train_labels.cuda(device=device)

			optimizer.zero_grad()

			train_outputs = model(train_inputs)
			train_loss = criterion(train_outputs, train_labels)
			train_loss.backward()
			optimizer.step()

			running_loss += train_loss.item()
			if display and i % 20 == 0:
				sys.stdout.write('\b' * 12 + ('%s' % (time.strftime('%H:%M:%S', time.gmtime()) + (' %3d' % i))))
		if display:
			sys.stdout.write('\b' * 12 + ('%s' % (time.strftime('%H:%M:%S', time.gmtime()) + (' %3d\n' % (i + 1)))))
			sys.stdout.write(time.strftime(' Eval : %H:%M:%S', time.gmtime()) + (' %3d ~ ' % len(eval_loader)) + (' ' * 12))
		eval_loss_sum = 0
		eval_acc_sum = 0
		for i, eval_data in enumerate(eval_loader):
			eval_inputs, eval_labels = eval_data
			if cuda:
				eval_inputs = eval_inputs.cuda(device=device)
				eval_labels = eval_labels.cuda(device=device)
			eval_outputs = model(eval_inputs).detach()
			eval_loss = criterion(eval_outputs, eval_labels)
			eval_pred = torch.argmax(eval_outputs, dim=1)
			eval_loss_sum += eval_loss
			eval_acc_sum += torch.sum(eval_pred == eval_labels)
			if display and i % 20 == 0:
				sys.stdout.write('\b' * 12 + ('%s' % (time.strftime('%H:%M:%S', time.gmtime()) + (' %3d' % i))))
		if display:
			sys.stdout.write('\b' * 12 + '%s' % (time.strftime('%H:%M:%S', time.gmtime()) + (' %3d\n' % (i + 1))))
		eval_loss_avg = eval_loss_sum.item() / float(len(eval_loader.sampler))
		eval_acc_avg = eval_acc_sum.item() / float(len(eval_loader.sampler))

		eval_acc_list.append(eval_acc_avg)

		if display:
			print('%4d epoch Train running loss: %.3f / Eval Avg. loss: %.3f / Eval Avg. Acc.: %5.2f%%'
			      % (epoch + 1, running_loss / 2000, eval_loss_avg, eval_acc_avg * 100))

		if len(eval_acc_list) > N_COMPARE:
			if [eval_acc_list[i] >= eval_acc_list[-N_COMPARE - 1] for i in range(-1, -N_COMPARE - 1, -1)].count(
					True) == 0:
				break
	return np.max(eval_acc_list)


def array2network(x, n_nodes):
	node_type = x[:2 * (n_nodes - 2)]
	connectivity = x[2 * (n_nodes - 2):]
	adj_mat = np.zeros((n_nodes, n_nodes))
	ind = 0
	for i in range(n_nodes):
		adj_mat[i, i + 1:] = connectivity[ind:ind + (n_nodes - i - 1)]
		ind = ind + (n_nodes - i - 1)
	adj_mat = valid_net_topo(adj_mat)
	return node_type, adj_mat


if __name__ == '__main__':
	parser_ = argparse.ArgumentParser('Simple NAS - NN training')
	parser_.add_argument('--data_type', dest='data_type', type=str)
	parser_.add_argument('--net_config', dest='net_config', type=str)
	parser_.add_argument('--n_nodes', dest='n_nodes', type=int)
	parser_.add_argument('--n_epochs', dest='n_epochs', type=int)
	parser_.add_argument('--n_ch_in', dest='n_ch_in', type=int)
	parser_.add_argument('--h_in', dest='h_in', type=int)
	parser_.add_argument('--w_in', dest='w_in', type=int)
	parser_.add_argument('--n_ch_base', dest='n_ch_base', type=int)
	parser_.add_argument('--device', dest='device', type=int, default=0)

	args_ = parser_.parse_args()

	data_type_ = args_.data_type
	net_config_ = args_.net_config
	n_nodes_ = args_.n_nodes
	n_epochs_ = args_.n_epochs
	n_ch_in_ = args_.n_ch_in
	h_in_ = args_.h_in
	w_in_ = args_.w_in
	n_ch_base_ = args_.n_ch_base
	device_ = args_.device

	n_edges_ = int(n_nodes_ * (n_nodes_ - 1) / 2)
	n_variables_ = int(n_edges_ + (n_nodes_ - 2) * 2)
	assert len(net_config_) == n_variables_
	node_type_, adj_mat_ = array2network(np.array([int(net_config_[i:i+1]) for i in range(n_variables_)]), n_nodes_)

	if adj_mat_ is None:
		eval_acc_ = 0.1
		flops_ = -1
	else:
		model_ = NASBinaryCNN(data_type_, node_type_, adj_mat_,
		                      n_ch_in=n_ch_in_, h_in=h_in_, w_in=w_in_, n_ch_base=n_ch_base_)
		if data_type_ == 'MNIST':
			train_loader_, valid_loader_, _ = load_mnist(batch_size=100, shuffle=True, random_seed=0)
		elif data_type_ == 'FashionMNIST':
			train_loader_, valid_loader_, _ = load_fashionmnist(batch_size=100, shuffle=True, random_seed=0)
		elif data_type_ == 'CIFAR10':
			train_loader_, valid_loader_, _ = load_cifar10(batch_size=100, shuffle=True, random_seed=0)
		eval_acc_ = train(model_, n_epochs_, train_loader_, valid_loader_, device_, display=False)
		dummy_input_ = next(model_.parameters()).data.ones_like(1, n_ch_in_, h_in_, w_in_)
		flops_ = count_ops(model_, dummy_input_)

	print('eval_acc:%.4f flops:%d' % (eval_acc_, flops_))