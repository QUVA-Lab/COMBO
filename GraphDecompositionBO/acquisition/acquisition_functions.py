import torch
from torch.distributions.normal import Normal


def expected_improvement(mean, var, reference):
	"""
	expected_improvement for minimization problems
	On graphs, we do not use a gradient-based optimization, thus backward does not have to be implemented
	:param mean: 
	:param var: 
	:param reference: 
	:return: 
	"""
	predictive_normal = Normal(mean.new_zeros(mean.size()), mean.new_ones(mean.size()))
	std = torch.sqrt(var)
	standardized = (-mean + reference) / std
	return (std * torch.exp(predictive_normal.log_prob(standardized)) + (-mean + reference) * predictive_normal.cdf(standardized)).clamp(min=0)