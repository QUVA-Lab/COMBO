import numpy as np
from scipy.special import gammaln

from GraphDecompositionBO.graphGP.sampler.tool_partition import compute_group_size

# For numerical stability in exponential
LOG_LOWER_BND = -12.0
LOG_UPPER_BND = 20.0
# For sampling stability
STABLE_MEAN_RNG = 1.0
# Hyperparameter for graph factorization
GRAPH_SIZE_LIMIT = 1024 + 2


def log_prior_constmean(constmean, output_min, output_max):
	"""
	:param constmean: numeric(float)
	:param output_min: numeric(float)
	:param output_max: numeric(float)
	:return:
	"""
	output_mid = (output_min + output_max) / 2.0
	output_rad = (output_max - output_min) * STABLE_MEAN_RNG / 2.0
	# Unstable parameter in sampling
	if constmean < output_mid - output_rad or output_mid + output_rad < constmean:
		return -float('inf')
	# Uniform prior
	# return 0
	# Truncated Gaussian
	stable_dev = output_rad / 2.0
	return -np.log(stable_dev) - 0.5 * (constmean - output_mid) ** 2 / stable_dev ** 2


def log_prior_noisevar(log_noise_var):
	if log_noise_var < LOG_LOWER_BND or min(LOG_UPPER_BND, np.log(10000.0)) < log_noise_var:
		return -float('inf')
	return np.log(np.log(1.0 + (0.1 / np.exp(log_noise_var)) ** 2))


def log_prior_kernelamp(log_amp, output_var, kernel_min, kernel_max):
	"""

	:param log_amp:
	:param output_var: numeric(float)
	:param kernel_min: numeric(float)
	:param kernel_max: numeric(float)
	:return:
	"""
	if log_amp < LOG_LOWER_BND or min(LOG_UPPER_BND, np.log(10000.0)) < log_amp:
		return -float('inf')
	log_amp_lower = np.log(output_var) - np.log(kernel_max)
	log_amp_upper = np.log(output_var) - np.log(max(kernel_min, 1e-100))
	log_amp_mid = 0.5 * (log_amp_upper + log_amp_lower)
	log_amp_rad = 0.5 * (log_amp_upper - log_amp_lower)
	log_amp_std = log_amp_rad / 2.0
	return -np.log(log_amp_std) - 0.5 * (log_amp - log_amp_mid) ** 2 / log_amp_std ** 2
	# return
	# Uniform
	# return 0 if kernel_min < output_var / amp < kernel_max else -float('inf')
	# Gamma
	# shape = output_var
	# rate = 1.0
	# return shape * np.log(rate) - gammaln(shape) + (shape - 1.0) * log_amp - rate * np.exp(log_amp)


def log_prior_edgeweight(log_beta_i):
	"""

	:param log_beta_i: numeric(float), ind-th element of 'log_beta'
	:param dim:
	:return:
	"""
	if log_beta_i < LOG_LOWER_BND or min(LOG_UPPER_BND, np.log(100.0)) < log_beta_i:
		return -float('inf')
	## Gamma prior For sparsity-inducing, shape should be 1
	## The higher the rate, the more sparsity is induced.
	# shape = 1.0
	# rate = 3.0
	# return shape * np.log(rate) - gammaln(shape) + (shape - 1.0) * log_beta_i - rate * np.exp(log_beta_i)

	## Horseshoe prior
	tau = 5.0
	return np.log(np.log(1.0 + 2.0 / (np.exp(log_beta_i) / tau) ** 2))

	## Laplace prior
	# scale = 0.5
	# return -np.exp(log_beta_i) / scale


def log_prior_partition(sorted_partition, n_vertices):
	"""
	Log of unnormalized density of given partition
	this prior prefers well-spread partition, which is quantified by induced entropy.
	Density is proportional to the entropy of a unnormalized probability vector consisting of [log(n_vertices in subgraph_i)]_i=1...N
	:param sorted_partition:
	:param n_vertices: 1D np.array
	:return:
	"""
	if len(sorted_partition) == 1 or compute_group_size(sorted_partition=sorted_partition, n_vertices=n_vertices) > GRAPH_SIZE_LIMIT:
		return -float('inf')
	else:
		prob_mass = np.array([np.sum(np.log(n_vertices[subset])) for subset in sorted_partition])
		prob_mass /= np.sum(prob_mass)
		entropy_mass = -np.sum(prob_mass * np.log(prob_mass))
		max_log = np.sum(np.log(n_vertices))
		thr_log = np.log(GRAPH_SIZE_LIMIT)
		n_chunk = int(np.floor(max_log / thr_log))
		prob_base = np.array([np.log(GRAPH_SIZE_LIMIT) for _ in range(n_chunk)] + [max_log - n_chunk * thr_log])
		prob_base /= np.sum(prob_base)
		entropy_base = -np.sum(prob_base * np.log(prob_base))
		return np.log(entropy_mass - entropy_base) * 5
		# return np.log(entropy_mass) * 5


if __name__ == '__main__':
	n_variables_ = 60
	n_vertices_ = np.ones((n_variables_, )) * 2
	sorted_partition_ = [[m_] for m_ in range(n_variables_)]
	print(sorted_partition_)
	print(np.exp(log_prior_partition(sorted_partition_, n_vertices_)))
	for _ in range(10):
		cnt_ = 0
		sorted_partition_ = []
		while cnt_ < n_variables_:
			prev_cnt_ = cnt_
			curr_cnt_ = cnt_ + np.random.randint(1, 3)
			sorted_partition_.append(list(range(prev_cnt_, min(curr_cnt_, n_variables_))))
			cnt_ = curr_cnt_
		print(sorted_partition_)
		print(np.exp(log_prior_partition(sorted_partition_, n_vertices_)))
	# import matplotlib.pyplot as plt_
	#
	# def marginal_(pdf):
	# 	x = np.linspace(0 + 1e-2, 10 + 1e-4, 100000)
	# 	y = pdf(x)
	# 	upper = y[:1]
	# 	lower = y[-1:]
	# 	x_grid = x[1:] - x[:-1]
	# 	return (np.sum(upper * x_grid) + np.sum(lower * x_grid)) / 2.0
	#
	# x_plot_ = np.linspace(1e-2, 2, 10000)
	# x_marginal_ = np.linspace(0, 10, 10000)
	# y_ip_ = lambda x: 1.0 / x
	# y_cc_ = lambda x, gamma: 1.0 / (np.pi * gamma * (1.0 + x / gamma) ** 2)
	# y_hs_ = lambda x, tau: (0.5 * np.log(1.0 + 4.0 / (x / tau) ** 2) + np.log(1.0 + 2.0 / (x / tau) ** 2)) / 2.0
	# C_ip_ = marginal_(y_ip_)
	# print(C_ip_)
	#
	# plt_.plot(x_plot_, y_ip_(x_plot_) / C_ip_, label='improper')
	# plt_.plot(x_plot_, y_cc_(x_plot_, 1.0), label='cauchy')
	# for tau in [1.0, 2.0, 5.0, 10.0]:
	# 	C_hs_ = marginal_(lambda x: y_hs_(x, tau))
	# 	print(C_hs_)
	# 	plt_.plot(x_plot_, y_hs_(x_plot_, tau) / C_hs_, label='HS %4.1f' % tau, alpha=0.25)
	# plt_.legend()
	# plt_.ylim([0, 0.2])
	# plt_.show()
