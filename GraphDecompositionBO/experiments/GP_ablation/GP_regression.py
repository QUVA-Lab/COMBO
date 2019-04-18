import argparse

import torch

from GraphDecompositionBO.graphGP.sampler.sample_hyper import slice_hyper
from GraphDecompositionBO.graphGP.sampler.sample_edgeweight import slice_edgeweight
from GraphDecompositionBO.graphGP.sampler.sample_partition import gibbs_partition

from GraphDecompositionBO.graphGP.models.gp_regression import GPRegression
from GraphDecompositionBO.graphGP.kernels.diffusionkernel import DiffusionKernel

from GraphDecompositionBO.experiments.GP_ablation.data_loader import load_highorderbinary


def GP_regression_posterior_sampling(model, train_input, train_output):
	n_variable = train_input.size()[1]
	categories = [2 for _ in range(n_variable)]
	list_of_adjacency = [torch.ones(2, 2) - torch.eye(2) for _ in range(n_variable)]
	log_beta = torch.zeros(n_variable)
	sorted_partition = [[c] for c in range(n_variable)]

	sample_hyper = []
	sample_log_beta = []
	sample_partition = []
	sample_freq = []
	sample_basis = []


if __name__ == '__main__':
	parser_ = argparse.ArgumentParser(description='GRASB : GRAph Signal Bayesian optimization')
	parser_.add_argument('-d', '--data_type', dest='data_type', type=int)
	parser_.add_argument('-s', '--data_scale', dest='train_data_scale', type=int)
	parser_.add_argument('-r', '--random_seed', dest='random_seed', type=int)
	args_ = parser_.parse_args()

	(train_input_, train_output_), (test_input_, test_output_) = load_highorderbinary(args_.data_type, args_.train_data_scale, random_seed=args_.random_seed)
	n_variables_ = train_input_.size(1)
	adjacency_mat_ = []
	fourier_freq_list_ = []
	fourier_basis_list_ = []
	for i_ in range(n_variables_):
		adjmat_ = torch.diag(torch.ones(1), -1) + torch.diag(torch.ones(1), 1)
		adjacency_mat_.append(adjmat_)
		laplacian_ = torch.diag(torch.sum(adjmat_, dim=0)) - adjmat_
		eigval_, eigvec_ = torch.symeig(laplacian_, eigenvectors=True)
		fourier_freq_list_.append(eigval_)
		fourier_basis_list_.append(eigvec_)
	kernel_ = DiffusionKernel(fourier_freq_list=fourier_freq_list_, fourier_basis_list=fourier_basis_list_)
	model_ = GPRegression(kernel=kernel_)