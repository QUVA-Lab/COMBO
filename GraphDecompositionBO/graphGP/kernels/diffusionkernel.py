import math

import torch
from GraphDecompositionBO.graphGP.kernels.graphkernel import GraphKernel


class DiffusionKernel(GraphKernel):
	"""
	Usually Graph Kernel means a kernel between graphs, here this kernel is a kernel between vertices on a graph
	Edge scales are not included in the module, instead edge weights of each subgraphs is used to calculate frequencies (fourier_freq)
	"""
	def __init__(self, fourier_freq_list, fourier_basis_list):
		super(DiffusionKernel, self).__init__(fourier_freq_list, fourier_basis_list)

	def forward(self, x1, x2=None, diagonal=False):
		"""
		:param x1, x2: each row is a vector with vertex numbers starting from 0 for each 
		:return: 
		"""
		if diagonal:
			assert x2 is None

		stabilizer = 0
		if x2 is None:
			x2 = x1
			if diagonal:
				stabilizer = 1e-6 * x1.new_ones(x1.size(0), 1, dtype=torch.float32)
			else:
				stabilizer = torch.diag(1e-6 * x1.new_ones(x1.size(0), dtype=torch.float32))

		full_gram = 1
		for i in range(len(self.fourier_freq_list)):
			fourier_freq = self.fourier_freq_list[i]
			fourier_basis = self.fourier_basis_list[i]

			subvec1 = fourier_basis[x1[:, i]]
			subvec2 = fourier_basis[x2[:, i]]
			freq_transform = torch.exp(-fourier_freq)

			if diagonal:
				factor_gram = torch.sum(subvec1 * freq_transform.unsqueeze(0) * subvec2, dim=1, keepdim=True)
			else:
				factor_gram = torch.matmul(subvec1 * freq_transform.unsqueeze(0), subvec2.t())
			# HACK for numerical stability for scalability
			full_gram *= factor_gram / torch.mean(freq_transform)

		return torch.exp(self.log_amp) * (full_gram + stabilizer)


if __name__ == '__main__':
	pass