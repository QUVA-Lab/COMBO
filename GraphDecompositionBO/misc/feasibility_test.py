import time
import sys
import copy
import numpy as np

import torch

from GraphDecompositionBO.graphGP.kernels.diffusionkernel import DiffusionKernel
from GraphDecompositionBO.graphGP.models.gp_regression import GPRegression
from GraphDecompositionBO.graphGP.inference.inference import Inference


if __name__ == '__main__':
	n_vars = 100
	n_data = 100
	n_evals = 100
	mat_size = 256
	symmat = torch.randn(mat_size, mat_size)
	symmat = torch.matmul(symmat, symmat.t()) + torch.eye(mat_size)
	start_time = time.time()
	for _ in range(n_evals):
		eigval, eigvec = torch.symeig(symmat, eigenvectors=True)
	end_time = time.time()
	print('Total time for %d eigen decomposition' % n_evals)
	print(end_time - start_time)
	print('Average time')
	print((end_time - start_time) / n_evals)