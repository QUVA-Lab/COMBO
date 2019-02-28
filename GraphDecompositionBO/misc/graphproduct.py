import numpy as np

BETA = 1.0

def frequency_transform(frequency):
	return np.exp(-BETA * frequency)


def fullkron(laplacian1, laplacian2, x1, x2=None):
	eye1 = np.eye(*laplacian1.shape)
	eye2 = np.eye(*laplacian2.shape)
	laplacian = np.kron(laplacian1, eye2) + np.kron(eye1, laplacian2)
	eigval, eigvec = np.linalg.eigh(laplacian)
	if x2 is None:
		x2 = x1
	v2 = laplacian2.shape[0]
	kron_x1 = x1[:, 0] * v2 + x1[:, 1]
	kron_x2 = x2[:, 0] * v2 + x2[:, 1]
	subvec1 = eigvec[kron_x1]
	subvec2 = eigvec[kron_x2]

	return np.dot(subvec1 * frequency_transform(eigval.reshape(-1, eigval.size)), subvec2.T)


def factorize(laplacian1, laplacian2, x1, x2=None):
	eigval1, eigvec1 = np.linalg.eigh(laplacian1)
	eigval2, eigvec2 = np.linalg.eigh(laplacian2)
	if x2 is None:
		x2 = x1
	gram1 = np.dot(eigvec1[x1[:, 0]] * frequency_transform(eigval1.reshape(-1, eigval1.size)), eigvec1[x2[:, 0]].T)
	gram2 = np.dot(eigvec2[x1[:, 1]] * frequency_transform(eigval2.reshape(-1, eigval2.size)), eigvec2[x2[:, 1]].T)
	return gram1 * gram2


if __name__ == '__main__':
	BETA = 0.2
	laplacian_list = []
	graph_size = []
	n_variables = 2
	n_data = 10
	for i in range(n_variables):
		n_v = int(np.random.randint(2, 21, (1,))[0])
		graph_size.append(n_v)
		# adjmat = np.tril(np.abs(np.random.normal(0, 1, (n_v, n_v))), -1)
		# adjmat = adjmat + adjmat.T
		# adjmat = np.diag(np.ones(n_v - 1), -1) + np.diag(np.ones(n_v - 1), 1)
		adjmat = np.ones((n_v, n_v)) - np.eye(n_v)
		laplacian = np.diag(np.sum(adjmat, axis=0)) - adjmat
		laplacian_list.append(laplacian)
	input_data = np.empty([0, n_variables])
	for i in range(n_data):
		datum = np.zeros([1, n_variables])
		for e, c in enumerate(graph_size):
			datum[0, e] = np.random.randint(0, c, (1,))[0]
		input_data = np.vstack([input_data, datum])
	input_data = input_data.astype(np.int)
	data1 = fullkron(laplacian_list[0], laplacian_list[1], input_data)
	data2 = factorize(laplacian_list[0], laplacian_list[1], input_data)
	assert np.isclose(data1, data2).all()
	for r in range(data2.shape[0]):
		print(' '.join(['%+.3E' % data2[r, i] for i in range(data2.shape[1])]))
	print(-np.log(frequency_transform(1)))

