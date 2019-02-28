import matplotlib.pyplot as plt

import numpy as np


def path_graph(n_v):
	adjmat = np.diag(np.ones(n_v - 1), -1) + np.diag(np.ones(n_v - 1), 1)
	laplacian = np.diag(np.sum(adjmat, axis=0)) - adjmat
	eigval, eigvec = np.linalg.eigh(laplacian)
	plt.hist(eigval)
	plt.title('Path graph with %d vertices / spectrum %.2E ~ %.2E' % (n_v, np.min(eigval), np.max(eigval)))
	plt.show()


if __name__ == '__main__':
	path_graph(10000)