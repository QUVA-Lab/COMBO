import numpy as np

import torch

from COMBO.experiments.test_functions import sample_init_points
from COMBO.experiments.test_functions.travel_plan import generate_travel_plan_problem, number_of_edges


TRAVEL_N_CITIES = 6
NORM_TIME = 20.0 * 10.0
NORM_DELAY = 2.0 * 10.0
NORM_COST = 200 * 10.0


def _time_waning(l):
	return 1.0 / (TRAVEL_N_CITIES - 1.0) - TRAVEL_N_CITIES / (TRAVEL_N_CITIES - 1.0) / l + 2


def _time_waxing(l):
	return (TRAVEL_N_CITIES - 2.0) / (TRAVEL_N_CITIES - 1.0) + TRAVEL_N_CITIES / (TRAVEL_N_CITIES - 1.0) / l


def _delay_min_max(transportation):
	# bus
	if transportation == 0:
		return 0.1, 0.25
	# train
	elif transportation == 1:
		return 0.25, 0.5
	# airplane
	elif transportation == 2:
		return 0.5, 2.5


def _cs_factors(curr_visit, adjusted_citywise_satisfaction, visit_history):
	"""
	adjusted_citywise_satisfaction: normally values lie between 70, 120
	thus cs_mean, cs_max, cs_prev are in the similar scale
	:param curr_visit:
	:param adjusted_citywise_satisfaction:
	:param visit_history:
	:return:
	"""
	cs_curr = adjusted_citywise_satisfaction[curr_visit - 1]
	if len(visit_history) > 0:
		cs_mean = cs_curr - np.mean(adjusted_citywise_satisfaction[np.array(visit_history) - 1])
		cs_max = cs_curr - np.max(adjusted_citywise_satisfaction[np.array(visit_history) - 1])
		cs_prev = cs_curr - adjusted_citywise_satisfaction[visit_history[-1] - 1]
	else:
		cs_mean = cs_curr - np.mean(adjusted_citywise_satisfaction)
		cs_max = cs_curr - np.max(adjusted_citywise_satisfaction)
		cs_prev = cs_curr - np.mean(adjusted_citywise_satisfaction)
	return cs_mean, cs_max, cs_prev


def _cs_to_fs(cs_mean, cs_max, cs_prev):
	return (0.25 * cs_mean + 0.25 * cs_max + 0.5 * cs_prev) * 0.02


def _tas_factors(curr_visit, tourism_attractions_similarity, visit_history, tas_mean_history):
	"""
	tourism_attractions_similarity : values are between [0, 1]
	tas_mean, tas_prev, tas_accum are in similar scale
	:param curr_visit:
	:param tourism_attractions_similarity:
	:param visit_history:
	:param tas_mean_history:
	:return:
	"""
	if len(visit_history) > 0:
		tas_mean = np.mean(tourism_attractions_similarity[curr_visit - 1][np.array(visit_history) - 1])
		tas_prev = tourism_attractions_similarity[curr_visit - 1][visit_history[-1] - 1]
		tas_accum = np.mean(tas_mean_history)
	else:
		tas_mean = 0.5
		tas_prev = 0.5
		tas_accum = 0.5
	return tas_mean, tas_prev, tas_accum


def _tas_adjust_cs(curr_visit, adjusted_citywise_satisfaction, tas_mean, tas_accum):
	"""
	RANDOM component should be scaled properly, not to make evaluation too noisy
	:param curr_visit:
	:param adjusted_citywise_satisfaction:
	:param tas_mean:
	:param tas_accum:
	:return:
	"""
	# this influences the stability and dynamics of the evaluation greatly
	randomness_scaling = 2.0
	cs_curr = adjusted_citywise_satisfaction[curr_visit - 1]
	return np.abs(np.random.normal(0, np.sqrt((1.0 - tas_mean) ** 2 + tas_accum ** 2))) * (2.0 * (np.random.uniform() < (cs_curr / 100.0)) - 1.0) * randomness_scaling


def _tas_to_fs(tas_mean, tas_prev):
	return 0.6 * (1.0 - tas_mean) + 0.4 * (1 - tas_prev)


def _tpt_next(curr_city, travel_choice, transportation_type, delay_prob):
	"""
	RANDOM component should be scaled properly, not to make evaluation too noisy
	:param x:
	:param curr_city:
	:param transportation_type:
	:param delay_prob:
	:return:
	"""
	ind = np.min(np.where(np.cumsum(transportation_type[:, curr_city - 1].flatten()) == travel_choice + 1)[0])
	transportation = ind // TRAVEL_N_CITIES
	next_city = ind % TRAVEL_N_CITIES + 1 # the city should be between 1 and TRAVEL_N_CITIES
	delay = 0
	if np.random.uniform() < delay_prob[transportation, curr_city - 1]:
		delay_min, delay_max = _delay_min_max(transportation)
		delay = delay_min + (delay_max - delay_min) * np.random.beta(0.5, 7.0 - transportation)
	return next_city, transportation, delay


def _tpt_factors(curr_city, next_city, transportation, delay, travel_time, cost, transporation_history, delay_history):
	if len(transporation_history) > 0:
		tpt_type = np.sum(np.array(delay_history)[np.array(transporation_history) == transportation]) / NORM_DELAY / float(len(transporation_history))
	else:
		tpt_type = 0
	tpt_delay = delay / NORM_DELAY
	tpt_cost = cost[transportation][curr_city - 1][next_city - 1] / NORM_COST * (1.0 + delay / NORM_DELAY)
	tpt_time = travel_time[transportation][curr_city - 1][next_city - 1] / NORM_TIME * (1.0 + delay / travel_time[transportation][curr_city - 1][next_city - 1])
	return tpt_type, tpt_delay, tpt_cost, tpt_time


def _tpt_to_fs(tpt_type, tpt_delay, tpt_cost, tpt_time):
	return 0.1 * tpt_type + 0.6 * tpt_delay + 0.2 * tpt_cost + 0.2 * tpt_time


def _tpt_adjust_cs(curr_city, next_city, transportation, travel_time, delay):
	adjtpt_dep = delay / NORM_DELAY
	adjtpt_des = travel_time[transportation][curr_city - 1][next_city - 1] / NORM_TIME
	return adjtpt_dep, adjtpt_des


def _compute_final_satisfaction(x, citywise_satisfaction, tourism_attractions_similarity, transportation_type, travel_time, cost, delay_prob):
	init_city = x[0]
	final_satisfaction = citywise_satisfaction[init_city - 1] / np.sum(citywise_satisfaction)
	adjusted_citywise_satisfaction = citywise_satisfaction.copy()
	visit_history = []
	tas_mean_history = []
	transporation_history = []
	delay_history = []
	curr_city = init_city
	for i in range(1, TRAVEL_N_CITIES + 1):
		satisfaction_here = 0
		# print('beginning %+6.4f, %+6.4f' % (satisfaction_here, final_satisfaction))

		# CS factors
		cs_mean, cs_max, cs_prev = _cs_factors(curr_city, adjusted_citywise_satisfaction, visit_history)
		# CS to FS
		satisfaction_here += _cs_to_fs(cs_mean, cs_max, cs_prev) * _time_waning(i + 1)
		# print('cs to fs  %+6.4f, %+6.4f' % (satisfaction_here, final_satisfaction))

		# TAS factors
		tas_mean, tas_prev, tas_accum = _tas_factors(curr_city, tourism_attractions_similarity, visit_history, tas_mean_history)
		# TAS to CS
		adj_tas = _tas_adjust_cs(curr_city, adjusted_citywise_satisfaction, tas_mean, tas_accum)
		adjusted_citywise_satisfaction[curr_city - 1] += adj_tas
		# TAS to FS
		satisfaction_here += _tas_to_fs(tas_mean, tas_prev) * _time_waning(i + 1)
		# print('fas to fs %+6.4f, %+6.4f' % (satisfaction_here, final_satisfaction))

		# TPT factors
		travel_choice = x[curr_city]
		next_city, transportation, delay = _tpt_next(curr_city, travel_choice, transportation_type, delay_prob)
		# TPT to CS
		adjtpt_dep, adjtpt_des = _tpt_adjust_cs(curr_city, next_city, transportation, travel_time, delay)
		adjusted_citywise_satisfaction[curr_city - 1] -= adjtpt_dep
		adjusted_citywise_satisfaction[next_city - 1] -= adjtpt_des
		# TPT to FS
		tpt_type, tpt_delay, tpt_cost, tpt_time = _tpt_factors(curr_city, next_city, transportation, delay, travel_time, cost, transporation_history, delay_history)
		satisfaction_here -= _tpt_to_fs(tpt_type, tpt_delay, tpt_cost, tpt_time) * _time_waxing(i + 1)
		# print('tpt_to_fs %+6.4f, %+6.4f' % (satisfaction_here, final_satisfaction))

		final_satisfaction += satisfaction_here
		# print('fs update        , %+6.4f' % (final_satisfaction))

		visit_history.append(curr_city)
		curr_city = next_city
		tas_mean_history.append(tas_mean)
		transporation_history.append(transportation)
		delay_history.append(delay)

	not_visited_penalty = (TRAVEL_N_CITIES - np.unique(visit_history).size) / float(TRAVEL_N_CITIES)
	cs_sum_bonus = (np.sum(adjusted_citywise_satisfaction[np.array(visit_history) - 1]) - np.sum(adjusted_citywise_satisfaction)) / float(TRAVEL_N_CITIES)
	final_adjustment = cs_sum_bonus * 0.1 - not_visited_penalty
	# print('penalty, bonus, %+6.4f, %6.4f' % (not_visited_penalty, cs_sum_bonus))
	# print('before final adjustment %+6.4f, %+6.4f' % (final_satisfaction, final_adjustment))
	final_satisfaction += final_adjustment
	# print(visit_history)
	# print('before', citywise_satisfaction)
	# print(' after', adjusted_citywise_satisfaction)
	# print('with final adjustment %+6.4f' % final_satisfaction)
	return final_satisfaction


class TravelPlan(object):
	"""
	Ising Sparsification Problem with the simplest graph
	"""
	def __init__(self, random_seed_pair=(None, None)):
		citywise_satisfaction, tourism_attractions_similarity, transportation_type, travel_time, cost, delay_prob = generate_travel_plan_problem(TRAVEL_N_CITIES, random_seed_pair[0])
		self.n_vertices = [TRAVEL_N_CITIES] + list(number_of_edges(transportation_type))
		self.suggested_init = torch.empty(0).long()
		self.suggested_init = torch.cat([self.suggested_init, sample_init_points(self.n_vertices, 20 - self.suggested_init.size(0), random_seed_pair[1]).long()], dim=0)
		self.adjacency_mat = []
		self.fourier_freq = []
		self.fourier_basis = []
		self.random_seed_info = 'R'.join([str(random_seed_pair[h]).zfill(4) if random_seed_pair[h] is not None else 'None' for h in range(2)])
		for i in range(len(self.n_vertices)):
			n_v = self.n_vertices[i]
			adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
			self.adjacency_mat.append(adjmat)
			laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
			eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
			self.fourier_freq.append(eigval)
			self.fourier_basis.append(eigvec)
		self.citywise_satisfaction = citywise_satisfaction
		self.tourism_attractions_similarity = tourism_attractions_similarity
		self.transportation_type = transportation_type
		self.travel_time = travel_time
		self.cost = cost
		self.delay_prob = delay_prob

	def evaluate(self, x):
		assert x.numel() == len(self.n_vertices)
		if x.dim() == 2:
			x = x.squeeze(0)
		x_np = (x.cpu() if x.is_cuda else x).numpy().copy()
		# starting city x_np[0] should be between 1 and TRAVEL_N_CITIES
		x_np[0] += 1
		assert np.sum([x_np[i + 1] + 1 > np.sum(self.transportation_type[:, i]) for i in range(TRAVEL_N_CITIES)]) == 0
		evaluation = _compute_final_satisfaction(x_np, self.citywise_satisfaction, self.tourism_attractions_similarity, self.transportation_type, self.travel_time, self.cost, self.delay_prob)
		# We need to maximize travelplan output but BO is currently set to do minimization, so -1 should be multiplied.
		return float(evaluation) * x.new_ones((1,)).float() * -1.0


if __name__ == '__main__':
	pass
	# check whether evalutation function is excessively noise or not
	# for p in range(10):
	# 	random_seeds = [1, 776, np.random.randint(0, 100000)]
	# 	evaluator = TravelPlan((random_seeds[0], random_seeds[1]))
	# 	print('-' * 30)
	# 	print('input seed %d' % random_seeds[2])
	# 	for q in range(20):
	# 		seeds_list = np.random.RandomState(random_seeds[2]).randint(0, 10000, (len(evaluator.n_vertices), ))
	# 		random_x = torch.Tensor([np.random.RandomState(seeds_list[i]).randint(0, evaluator.n_vertices[i]) for i in range(len(evaluator.n_vertices))]).long()
	# 		print(evaluator.evaluate(random_x))

	# check random search performance
	# evaluator = TravelPlan((6158, 7947))
	# n_evals = 10000
	# for _ in range(10):
	# 	lowest_negative_satisfaction = float('inf')
	# 	for i in range(n_evals):
	# 		if i < evaluator.suggested_init.size(0):
	# 			random_x = evaluator.suggested_init[i]
	# 		else:
	# 			random_x = torch.Tensor([np.random.randint(0, evaluator.n_vertices[h]) for h in range(len(evaluator.n_vertices))]).long()
	# 		negative_satisfaction = evaluator.evaluate(random_x).item()
	# 		if negative_satisfaction < lowest_negative_satisfaction:
	# 			lowest_negative_satisfaction = negative_satisfaction
	# 	print('With %d random search, the highest satisfaction is %f' % (n_evals, lowest_negative_satisfaction))

	evaluator = PestControl(5355)
	# x = np.random.RandomState(123).randint(0, 5, (PESTCONTROL_N_STAGES, ))
	# print(_pest_control_score(x))
	n_evals = 2000
	for _ in range(10):
		best_pest_control_loss = float('inf')
		for i in range(n_evals):
			if i < evaluator.suggested_init.size(0):
				random_x = evaluator.suggested_init[i]
			else:
				random_x = torch.Tensor([np.random.randint(0, 5) for h in range(len(evaluator.n_vertices))]).long()
			pest_control_loss = evaluator.evaluate(random_x).item()
			if pest_control_loss < best_pest_control_loss:
				best_pest_control_loss = pest_control_loss
		print('With %d random search, the pest control objective(%d stages) is %f' % (n_evals, PESTCONTROL_N_STAGES, best_pest_control_loss))

