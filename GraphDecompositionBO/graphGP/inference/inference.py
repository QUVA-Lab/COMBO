import time

import numpy as np
import sampyl as smp

import torch
import torch.nn as nn


class Inference(nn.Module):

	def __init__(self, train_data, model):
		super(Inference, self).__init__()
		self.model = model
		self.train_x = train_data[0]
		self.train_y = train_data[1]
		self.output_min = torch.min(self.train_y.data)
		self.output_max = torch.max(self.train_y.data)
		self.mean_vec = None
		self.gram_mat = None
		self.cholesky = None
		self.jitter = 0

	def reset_parameters(self):
		self.model.reset_parameters()

	def init_parameters(self):
		amp = float(torch.std(self.train_y))
		self.model.kernel.init_parameters(amp)
		self.model.mean.const_mean.data.fill_(torch.mean(self.train_y.data))
		self.model.likelihood.log_noise_var.data.fill_(np.log(amp / 1000.0))

	def stable_parameters(self):
		const_mean = float(self.model.mean.const_mean.data)
		return self.output_min <= const_mean <= self.output_max

	def log_kernel_amp(self):
		return self.model.log_kernel_amp()

	def gram_mat_update(self, hyper=None):
		if hyper is not None:
			self.model.vec_to_param(hyper)
		self.mean_vec = self.train_y - self.model.mean(self.train_x.float())
		self.gram_mat = self.model.kernel(self.train_x) + torch.diag(self.model.likelihood(self.train_x.float()))

	def cholesky_update(self, hyper):
		self.gram_mat_update(hyper)
		eye_mat = torch.diag(self.gram_mat.new_ones(self.gram_mat.size(0)))
		chol_jitter = 0
		while True:
			try:
				self.cholesky = torch.cholesky(self.gram_mat + eye_mat * chol_jitter, False)
				torch.gesv(self.gram_mat[:, :1], self.cholesky)
				break
			except RuntimeError:
				chol_jitter = self.gram_mat.data[0, 0] * 1e-6 if chol_jitter == 0 else chol_jitter * 10
		self.jitter = chol_jitter

	def predict(self, pred_x, hyper=None, verbose=False):
		if hyper is not None:
			param_original = self.model.param_to_vec()
			self.cholesky_update(hyper)

		k_pred_train = self.model.kernel(pred_x, self.train_x)
		k_pred = self.model.kernel(pred_x, diagonal=True)

		chol_solver = torch.gesv(torch.cat([k_pred_train.t(), self.mean_vec], 1), self.cholesky)[0]
		chol_solve_k = chol_solver[:, :-1]
		chol_solve_y = chol_solver[:, -1:]

		pred_mean = torch.mm(chol_solve_k.t(), chol_solve_y) + self.model.mean(pred_x)
		pred_quad = (chol_solve_k ** 2).sum(0).view(-1, 1)
		pred_var = k_pred - pred_quad

		if verbose:
			numerically_stable = (pred_var.data >= 0).all()
			zero_pred_var = (pred_var.data <= 0).all()

		if hyper is not None:
			self.cholesky_update(param_original)

		if verbose:
			return pred_mean, pred_var.clamp(min=1e-8), numerically_stable, zero_pred_var
		else:
			return pred_mean, pred_var.clamp(min=1e-8)

	def negative_log_likelihood(self, hyper=None):
		if hyper is not None:
			param_original = self.model.param_to_vec()
			self.cholesky_update(hyper)
		mean_vec_sol = torch.gesv(self.mean_vec, self.cholesky)[0]
		nll = 0.5 * torch.sum(mean_vec_sol ** 2) + torch.sum(torch.log(torch.diag(self.cholesky))) + 0.5 * self.train_y.size(0) * np.log(2 * np.pi)
		if hyper is not None:
			self.cholesky_update(param_original)
		return nll

	def sampling(self, n_sample=10, n_burnin=100, n_thin=10):
		type_as_arg = list(self.model.likelihood.parameters())[0].data
		def logp(hyper):
			hyper_tensor = torch.from_numpy(hyper).type_as(type_as_arg)
			if self.model.out_of_bounds(hyper):
				return -np.inf
			self.model.vec_to_param(hyper_tensor)
			if not self.stable_parameters():
				return -np.inf
			prior_ll = self.model.prior_log_lik(hyper)
			log_likelihood = -self.negative_log_likelihood(hyper_tensor).item()
			return prior_ll + log_likelihood
		# Sampling is continued from the parameter values from previous acquisition step
		hyper_numpy = self.model.param_to_vec().numpy()

		start_time = time.time()
		###--------------------------------------------------###
		# This block can be modified to use other sampling method
		sampler = smp.Slice(logp=logp, start={u'hyper': hyper_numpy}, compwise=True)
		samples = sampler.sample(n_burnin + n_thin * n_sample, burn=n_burnin + n_thin - 1, thin=n_thin)
		###--------------------------------------------------###
		print('Sampling : ' + time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))

		# Here, model parameters are updated and stored to model
		self.cholesky_update(torch.from_numpy(samples[-1][0]).type_as(type_as_arg))
		return torch.stack([torch.from_numpy(elm[0]) for elm in samples], 0).type_as(type_as_arg)


if __name__ == '__main__':
	pass