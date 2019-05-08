import os
import sys
import time
import GPUtil
import subprocess
import numpy as np
from pthflops import count_ops

import torch
import torch.cuda

from GraphDecompositionBO.experiments.NAS_binary.generate_architecture import NASBinaryCNN
from GraphDecompositionBO.experiments.NAS_binary.data_config import MNIST_N_CH_IN, MNIST_H_IN, MNIST_W_IN
from GraphDecompositionBO.experiments.NAS_binary.data_config import FashionMNIST_N_CH_IN, FashionMNIST_H_IN, FashionMNIST_W_IN
from GraphDecompositionBO.experiments.NAS_binary.data_config import CIFAR10_N_CH_IN, CIFAR10_H_IN, CIFAR10_W_IN

from GraphDecompositionBO.experiments.NAS_binary.architectures_in_binary import init_architectures


class NASBinary(object):
	def __init__(self, data_type, device=None):
		assert data_type in ['MNIST', 'FashionMNIST', 'CIFAR10']
		self.data_type = data_type
		self.n_nodes = 7
		self.n_edges = int(self.n_nodes * (self.n_nodes - 1) / 2)
		self.n_variables = int(self.n_edges + (self.n_nodes - 2) * 2)
		self.n_ch_base = 8
		self.n_epochs = 20
		self.device = device
		self.n_repeat = 3
		if torch.cuda.is_available():
			if len(GPUtil.getGPUs()) == 1:
				self.device = 0
			else:
				assert 0 <= self.device < len(GPUtil.getGPUs())
		else:
			self.device = None

		if self.data_type == 'MNIST':
			self.n_ch_in, self.h_in, self.w_in = MNIST_N_CH_IN, MNIST_H_IN, MNIST_W_IN
		elif self.data_type == 'FashionMNIST':
			self.n_ch_in, self.h_in, self.w_in = FashionMNIST_N_CH_IN, FashionMNIST_H_IN, FashionMNIST_W_IN
		elif self.data_type == 'CIFAR10':
			self.n_ch_in, self.h_in, self.w_in = CIFAR10_N_CH_IN, CIFAR10_H_IN, CIFAR10_W_IN

		self.n_vertices = np.array([2] * self.n_variables)

		most_complex_model = NASBinaryCNN(data_type, np.ones(2 * (self.n_nodes - 2)),
		                                  np.triu(np.ones((self.n_nodes, self.n_nodes)), 1),
		                                  n_ch_in=self.n_ch_in, h_in=self.h_in, w_in=self.w_in, n_ch_base=self.n_ch_base)

		self.suggested_init = init_architectures()
		dummy_input = next(most_complex_model.parameters()).new_ones(1, self.n_ch_in, self.h_in, self.w_in)
		self.max_flops = count_ops(most_complex_model, dummy_input)

		self.adjacency_mat = []
		self.fourier_freq = []
		self.fourier_basis = []
		for i in range(self.n_variables):
			adjmat = torch.diag(torch.ones(1), -1) + torch.diag(torch.ones(1), 1)
			self.adjacency_mat.append(adjmat)
			laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
			eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
			self.fourier_freq.append(eigval)
			self.fourier_basis.append(eigvec)

	def evaluate(self, x):
		if x.dim() == 1:
			x = x.unsqueeze(0)
		x = x.int()
		return torch.cat([self._evaluate_single(x[i]) for i in range(x.size(0))], dim=0)

	def _evaluate_single(self, x):
		assert x.numel() == self.n_variables
		assert x.dim() == 1
		command = self._generate_cmd(x)
		start_time = time.time()
		processes = [subprocess.Popen(command, stdout=subprocess.PIPE) for _ in range(self.n_repeat)]
		for p in processes:
			p.wait()
		stdout_read_list = [p.stdout.read() for p in processes]
		print(time.strftime('Time for training : %H:%M:%S', time.gmtime(time.time() - start_time)))
		results = [self._parse_stdout(stdout_read.decode('ascii').split('\n')[2]) for stdout_read in stdout_read_list]
		eval_acc, flops = zip(*[(elm['eval_acc'], elm['flops']) for elm in results])
		print(' '.join(['%6.4f' % (1.0 - elm) for elm in eval_acc]))
		eval_acc_mean, flops = np.mean(eval_acc), np.mean(flops)
		eval_err_mean = 1.0 - eval_acc_mean
		eval_std = np.std(eval_acc)
		flop_ratio = float(flops) / self.max_flops if flops >= 0 else 1.0
		const = 0.02
		eval = eval_err_mean + const * flop_ratio
		print('Err:%5.2f%% FLOPs:%6.4f(%4.2f)' % (eval_err_mean * 100, flop_ratio, const))
		return eval * x.float().new_ones(1, 1)

	def _generate_cmd(self, x):
		cmd_list = ['python', os.path.join(os.path.split(__file__)[0], 'nas_evaluation.py')]
		cmd_list += ['--data_type',  self.data_type]
		cmd_list += ['--net_config', ''.join([str(int(x[i].item())) for i in range(self.n_variables)])]
		cmd_list += ['--n_nodes', str(self.n_nodes)]
		cmd_list += ['--n_epochs', str(self.n_epochs)]
		cmd_list += ['--n_ch_in', str(self.n_ch_in)]
		cmd_list += ['--h_in', str(self.h_in)]
		cmd_list += ['--w_in', str(self.w_in)]
		cmd_list += ['--n_ch_base', str(self.n_ch_base)]
		cmd_list += ['--device', str(self.device)]
		return cmd_list

	@staticmethod
	def _parse_stdout(stdout_str):
		return {elm.split(':')[0]: float(elm.split(':')[1]) for elm in stdout_str.split(' ')}


if __name__ == '__main__':
	nas_binary_ = NASBinary(data_type='FashionMNIST', device=int(sys.argv[1]))
	x_ = torch.randint(1, 2, (nas_binary_.n_variables,))
	eval_list_ = []
	print(time.strftime('%H:%M:%S', time.gmtime()))
	for _ in range(1):
		eval_list_.append(nas_binary_.evaluate(x_).item())
		print(time.strftime('%H:%M:%S', time.gmtime()))
	print(eval_list_)
	print(np.mean(eval_list_))
	print(np.std(eval_list_))

