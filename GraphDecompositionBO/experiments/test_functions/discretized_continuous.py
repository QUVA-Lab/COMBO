import math
import numpy as np

import torch


class Branin(object):

	def __init__(self):
		self.n_vertices = np.array([51, 51])
		self.n_factors = len(self.n_vertices)
		self.suggested_init = torch.Tensor(self.n_vertices).long().unsqueeze(0) / 2
		for _ in range(1, 2):
			random_init = torch.cat([torch.randint(0, int(elm), (1, 1)) for elm in self.n_vertices], dim=1)
			self.suggested_init = torch.cat([self.suggested_init, random_init], dim=0)
		self.adjacency_mat = []
		self.fourier_freq = []
		self.fourier_basis = []
		for i in range(len(self.n_vertices)):
			n_v = self.n_vertices[i]
			adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
			adjmat *= (n_v - 1.0)
			self.adjacency_mat.append(adjmat)
			degmat = torch.sum(adjmat, dim=0)
			laplacian = (torch.diag(degmat) - adjmat)
			eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
			self.fourier_freq.append(eigval)
			self.fourier_basis.append(eigvec)

	def evaluate(self, x_g):
		flat = x_g.dim() == 1
		if flat:
			x_g = x_g.view(1, -1)
		ndim = x_g.size(1)
		assert ndim == len(self.n_vertices)
		n_repeat = int(ndim / 2)
		n_dummy = int(ndim % 2)

		x_e = torch.ones(x_g.size())
		for d in range(len(self.n_vertices)):
			x_e[:, d] = torch.linspace(-1, 1, int(self.n_vertices[d]))[x_g[:, d].long()]

		shift = torch.cat([torch.FloatTensor([2.5, 7.5]).repeat(n_repeat), torch.zeros(n_dummy)])

		x_e = x_e * 7.5 + shift

		a = 1
		b = 5.1 / (4 * math.pi ** 2)
		c = 5.0 / math.pi
		r = 6
		s = 10
		t = 1.0 / (8 * math.pi)
		output = 0
		for i in range(n_repeat):
			output += a * (x_e[:, 2 * i + 1] - b * x_e[:, 2 * i] ** 2 + c * x_e[:, 2 * i] - r) ** 2 \
			          + s * (1 - t) * torch.cos(x_e[:, 2 * i]) + s
		output /= float(n_repeat)
		if flat:
			return output.squeeze(0)
		else:
			return output


class Hartmann6(object):

	def __init__(self):
		self.n_vertices = np.array([51] * 6)
		self.n_factors = len(self.n_vertices)
		self.suggested_init = torch.Tensor(self.n_vertices).long().unsqueeze(0) / 2
		for _ in range(1, 2):
			random_init = torch.cat([torch.randint(0, int(elm), (1, 1)) for elm in self.n_vertices], dim=1)
			self.suggested_init = torch.cat([self.suggested_init, random_init], dim=0)
		self.adjacency_mat = []
		self.fourier_freq = []
		self.fourier_basis = []
		for i in range(len(self.n_vertices)):
			n_v = self.n_vertices[i]
			adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
			adjmat *= (n_v - 1.0)
			self.adjacency_mat.append(adjmat)
			wgtsum = torch.sum(adjmat, dim=0)
			laplacian = (torch.diag(wgtsum) - adjmat)
			eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
			self.fourier_freq.append(eigval)
			self.fourier_basis.append(eigvec)

	def evaluate(self, x_g):
		alpha = torch.FloatTensor([1.0, 1.2, 3.0, 3.2])
		A = torch.FloatTensor([[10.0, 3.00, 17.0, 3.50, 1.70, 8.00],
							   [0.05, 10.0, 17.0, 0.10, 8.00, 14.0],
							   [3.00, 3.50, 1.70, 10.0, 17.0, 8.00],
							   [17.0, 8.00, 0.05, 10.0, 0.10, 14.0]]).t()
		P = torch.FloatTensor([[0.1312,0.1696,0.5569,0.0124,0.8283,0.5886],
							   [0.2329,0.4135,0.8307,0.3736,0.1004,0.9991],
							   [0.2348,0.1451,0.3522,0.2883,0.3047,0.6650],
							   [0.4047,0.8828,0.8732,0.5743,0.1091,0.0381]]).t()

		flat = x_g.dim() == 1
		if flat:
			x_g = x_g.view(1, -1)
		ndata, ndim = x_g.size()
		n_repeat = int(ndim / 6)

		x_e = torch.ones(x_g.size())
		for d in range(len(self.n_vertices)):
			x_e[:, d] = torch.linspace(-1, 1, int(self.n_vertices[d]))[x_g[:, d]]

		x_e = (x_e + 1) * 0.5

		output = 0
		for i in range(n_repeat):
			x_block = x_e[:, 6 * i:6 * (i + 1)]
			output += -(alpha.view(1, -1).repeat(ndata, 1)
			            * torch.exp(-(A.unsqueeze(0).repeat(ndata, 1, 1)
			                          * (
					                          x_block.unsqueeze(2).repeat(1, 1, 4)
			                             - P.unsqueeze(0).repeat(ndata, 1, 1)) ** 2).sum(1))).sum(1, keepdim=True)
		output /= float(n_repeat)
		if flat:
			return output.squeeze(0)
		else:
			return output


if __name__ == '__main__':
	from itertools import product

	objective = Branin()

	minimum = float('inf')
	grid_inputs = torch.Tensor(list(product(*[range(objective.n_vertices[i]) for i in range(len(objective.n_vertices))])))
	sorted_eval, _ = torch.sort(objective.evaluate(grid_inputs.long()).squeeze(), descending=False)
	print(sorted_eval[:30])
