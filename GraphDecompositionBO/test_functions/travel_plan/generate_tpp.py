import numpy as np

# 0: bus, 1: train, 2:airplane
N_TRANSPORTATION = 3


def number_of_edges(transportation_type):
	return np.sum(transportation_type, axis=2).sum(axis=0).astype(np.int)


def generate_travel_plan_problem(n_cities, random_seed=None):
	seeds_list = np.random.RandomState(random_seed).randint(0, 10000, (3,))
	transportation_type, travel_time, cost, delay_prob = _generate_transportation_config(n_cities, seeds_list[0])
	citywise_satisfaction = _generate_citywise_satisfaction(n_cities, seeds_list[1])
	tourism_attractions_similarity = _generate_tourism_attraction_similarities(n_cities, seeds_list[2])
	return citywise_satisfaction, tourism_attractions_similarity, transportation_type, travel_time, cost, delay_prob


def _generate_transportation_config(n_cities, random_seed=None):
	seeds_list = np.random.RandomState(random_seed).randint(0, 10000, (10, ))
	transportation = np.zeros((N_TRANSPORTATION, n_cities, n_cities))
	distance = np.random.RandomState(seeds_list[0]).randint(300, 3000, (n_cities, n_cities)).astype(np.float)
	# distance is symmetric
	for i in range(n_cities):
		distance[i][i] = 0
		for j in range(i, n_cities):
			distance[i][j] = distance[j][i]
	travel_time = np.zeros((N_TRANSPORTATION, n_cities, n_cities))

	# bus travel time for given distance
	travel_time[0] = np.round(distance, -1) * 0.01
	# bus travel time can be a lot different from straight line distance
	travel_time[0] += np.round(np.abs(np.random.RandomState(seeds_list[1]).normal(0, travel_time[0] * 0.2)), 2)
	travel_time[0] = np.round(travel_time[0] / 0.1) * 0.1

	# train travel time is slightly shorter than bus
	travel_time[1] = np.round(distance, -1) * 0.009
	# train travel time can be
	travel_time[1] += np.round(np.abs(np.random.RandomState(seeds_list[2]).normal(0, travel_time[1] * 0.2)), 2)
	travel_time[1] = np.round(travel_time[1] / 0.1) * 0.1

	# airplane travel time
	travel_time[2] = np.round(distance, -1) * 0.002
	travel_time[2] += np.round(np.abs(np.random.RandomState(seeds_list[3]).normal(0, travel_time[2] * 0.05)), 1)
	travel_time[2] = np.round(travel_time[2] / 0.1) * 0.1

	for t in range(travel_time.shape[0]):
		for i in range(n_cities):
			travel_time[t][i][i] = 0
			for j in range(i, n_cities):
				travel_time[t][i][j] = travel_time[t][j][i]

	cost = np.zeros((N_TRANSPORTATION, n_cities, n_cities))
	# bus cost
	cost[0] = np.round(distance, -1) * 0.05
	cost[0] += 0.5 * travel_time[0]
	cost[0] = np.round(cost[0])

	# train cost
	cost[1] = np.round(distance, -1) * 0.05 * 1.3
	cost[1] += 0.75 * travel_time[1]
	cost[1] = np.round(cost[1])

	# airplane cost
	cost[2] = 30 + np.round(distance, -1) * 0.1
	cost[2] = np.round(cost[2])

	for t in range(cost.shape[0]):
		for i in range(n_cities):
			cost[t][i][i] = 0
			for j in range(i, n_cities):
				cost[t][i][j] = cost[t][j][i]

	transportation[0, :, :] = 1
	transportation[1, :, :] = np.abs(np.random.RandomState(seeds_list[4]).normal(0, np.abs(distance - 1500))) < 200
	transportation[2, :, :] = np.random.RandomState(seeds_list[5]).uniform(0, 1, (n_cities, n_cities)) > 0.5

	for t in range(transportation.shape[0]):
		for i in range(n_cities):
			transportation[t][i][i] = 0
			for j in range(n_cities):
				transportation[t][i][j] = max(transportation[t][j][i], transportation[t][i][j])
	for i in range(n_cities):
		for j in range(n_cities):
			if transportation[1][i][j] == 0:
				transportation[2][i][j] = 1
		transportation[2][i][i] = 0

	delay_prob = np.zeros((N_TRANSPORTATION, n_cities))
	delay_prob[0, :] = 0.1
	delay_prob[1, :] = 0.1 + np.round(0.2 * np.sum(transportation[1, :, :], axis=0) / float(n_cities - 1), 1)
	delay_prob[2, :] = 0.1 + np.round(0.4 * np.sum(transportation[2, :, :], axis=0) / float(n_cities - 1), 1)

	return transportation, travel_time, cost, delay_prob


def _generate_citywise_satisfaction(n_cities, random_seed=None):
	return np.random.RandomState(random_seed).randint(75, 95, (n_cities, ))


def _generate_tourism_attraction_similarities(n_cities, random_seed=None):
	sim_mat = np.random.RandomState(random_seed).uniform(0.2, 1.0, (n_cities, n_cities))
	for i in range(n_cities):
		sim_mat[i][i] = 1.0
		for j in range(i, n_cities):
			sim_mat[i][j] = sim_mat[j][i]
	return sim_mat


if __name__ == '__main__':
	citywise_satisfaction, tourism_attractions_similarity, transportation_type, travel_time, cost, delay_prob = generate_travel_plan_problem(5, 0)
	print(delay_prob)
	print(number_of_edges(transportation_type))