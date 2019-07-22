import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


CONTAMINATION_PYTHON_MAT_DIR = '/home/coh1/Experiments/COMBO/Contamination_Others_python_friendly_mat_files'
CONTAMINATION_GRASP_DATA_DIR = '/home/coh1/Experiments/COMBO/Contamination_COMBO'
ISING_PYTHON_MAT_DIR = '/home/coh1/Experiments/COMBO/Ising_Others_python_friendly_mat_files'
ISING_GRASP_DATA_DIR = '/home/coh1/Experiments/COMBO/Ising_COMBO'
AERO_PYTHON_MAT_DIR = '/home/coh1/Experiments/COMBO/Aero_Others_python_friendly_mat_files'
AERO_GRASP_DATA_DIR = '/home/coh1/Experiments/COMBO/Aero_COMBO'


def name_in_plot(name):
	if 'COMBO' in name:
		return name
	elif name == 'BOCSorder2SA':
		return 'Bayes-SA'
	elif name == 'BOCSorder2SDP':
		return 'Bayes-SDP'
	elif name == 'ExpectedImprovement':
		return 'EI'
	elif name == 'HorseShoeorder2SA':
		return 'BOCS-SA'
	elif name == 'HorseShoeorder2SDP':
		return 'BOCS-SDP'
	elif name == 'MLEorder2SA':
		return 'MLE-SA'
	elif name == 'MLEorder2SDP':
		return 'MLE-SDP'
	elif name == 'ObliviousLocalSearch':
		return 'OLS'
	elif name == 'RandomSearch':
		return 'RS'
	elif name == 'SMAC':
		return 'SMAC'
	elif name == 'SequentialMonteCarlo':
		return 'PS'
	elif name == 'SimulatedAnnealing':
		return 'SA'
	else:
		raise ValueError('Not supported algorithm name')


def assign_color(model_name):
	if model_name == 'COMBO':
		return 'k'
	elif model_name == 'SMAC':
		return 'm'
	elif model_name == 'SimulatedAnnealing':
		return 'g'
	elif model_name == 'RandomSearch':
		return 'r'
	return np.random.RandomState(sum([ord(elm) + 1 for elm in model_name]) + 2).rand(3)


def collect_matlab_data(data_dir, lamdas=[0, 0.0001, 0.01]):
	mean_std_dict = dict(zip(lamdas, [{} for _ in range(len(lamdas))]))
	for elm in os.listdir(data_dir):
		problem_type, algorithm_type = elm.split('_')
		lamda = float(algorithm_type[-9:-4])
		algorithm_type = algorithm_type[:-9]
		matlab_data = sio.loadmat(os.path.join(data_dir, elm))['evaluation']
		mean_std_dict[lamda][algorithm_type] = (np.mean(matlab_data, axis=1), np.std(matlab_data, axis=1))
	return mean_std_dict


def collection_python_data(data_dir, identifier, lamdas=[0, 0.0001, 0.01]):
	directory_grouping = dict(zip(lamdas, [[] for _ in range(len(lamdas))]))
	for elm in os.listdir(data_dir):
		if identifier in elm:
			directory_grouping[float(elm.split('_')[2])].append(elm)
	data_grouping = dict(zip(lamdas, [{} for _ in range(len(lamdas))]))
	for key, value in directory_grouping.iteritems():
		evaluations = directory_python_data(os.path.join(data_dir, value[0]))
		for i in range(1, len(value)):
			evaluation_sample = directory_python_data(os.path.join(data_dir, value[i]))
			if evaluations.shape[0] > evaluation_sample.shape[0]:
				print('%d vs %d when stacking %s' % (evaluations.shape[0], evaluation_sample.shape[0], value[i]))
				evaluations = np.hstack([evaluations, np.vstack((evaluation_sample, np.zeros(evaluations.shape[0] - evaluation_sample.shape[0], 1)))])
			elif evaluation_sample.shape[0] > evaluations.shape[0]:
				print('%d vs %d when stacking %s' % (evaluations.shape[0], evaluation_sample.shape[0], value[i]))
				evaluations = np.hstack([np.vstack((evaluations, np.zeros(evaluation_sample.shape[0] - evaluations.shape[0], evaluations.shape[1]))), evaluations])
			else:
				evaluations = np.hstack([evaluations, evaluation_sample])
		data_grouping[key] = evaluations
	mean_std_dict = {}
	for key, value in data_grouping.iteritems():
		mean_std_dict[key] = np.mean(value, axis=1), np.std(value, axis=1)
	return mean_std_dict


def directory_python_data(data_dir):
	last_log_filename = str(max([int(elm[:4]) for elm in os.listdir(os.path.join(data_dir, 'log'))])).zfill(4) + '.out'
	last_log_file = open(os.path.join(data_dir, 'log', last_log_filename))
	log_lines = last_log_file.readlines()
	last_log_file.close()
	return np.array([objective_from_log_line(log_lines[i]) for i in range(len(log_lines))]).reshape(-1, 1)


def objective_from_log_line(log_line):
	return float(log_line.split()[15].split('(')[0])


def contamination_data():
	mean_std_dict = collect_matlab_data(CONTAMINATION_PYTHON_MAT_DIR, lamdas=[0, 0.0001, 0.01])
	graph_data = collection_python_data(CONTAMINATION_GRASP_DATA_DIR, identifier='Contamination1', lamdas=[0, 0.0001, 0.01])
	for key, value in mean_std_dict.iteritems():
		mean_std_dict[key]['COMBO'] = graph_data[key]
	return mean_std_dict


def ising_data():
	mean_std_dict = collect_matlab_data(ISING_PYTHON_MAT_DIR, lamdas=[0, 0.0001, 0.01])
	graph_data = collection_python_data(ISING_GRASP_DATA_DIR, identifier='Ising1', lamdas=[0, 0.0001, 0.01])
	for key, value in mean_std_dict.iteritems():
		mean_std_dict[key]['COMBO'] = graph_data[key]
	return mean_std_dict


def aero_data():
	mean_std_dict = collect_matlab_data(AERO_PYTHON_MAT_DIR, lamdas=[0.01])
	for i in range(1, 4):
		graph_data = collection_python_data(AERO_GRASP_DATA_DIR, identifier='AeroStruct' + str(i), lamdas=[0.01])
		for key, value in mean_std_dict.iteritems():
			mean_std_dict[key]['COMBO' + str(i)] = graph_data[key]
	return mean_std_dict


def plotting_mean_std(mean_std_dict, lamda, method_type):
	assert lamda in [0, 0.0001, 0.01]
	begin_ind = 20
	z_value = 1.0 / 25.0 ** 0.5
	model_str = ''
	mean_std_str = ''
	algorithm_list = ['COMBO', 'ExpectedImprovement', 'HorseShoeorder2SDP', 'ObliviousLocalSearch', 'RandomSearch', 'SMAC', 'SimulatedAnnealing']
	for key in sorted(mean_std_dict[lamda].keys()):
		if key in algorithm_list or 'COMBO' in key:
			value = mean_std_dict[lamda][key]
			x_coord = range(begin_ind + 1, value[0].size + 1)
			mean = value[0][begin_ind:]
			std = value[1][begin_ind:]
			color = assign_color(key)
			short_name = name_in_plot(key)
			plt.plot(x_coord, mean, label=short_name, color=color)
			plt.fill_between(x_coord, mean - z_value * std, mean + z_value * std, color=color, alpha=0.25)
			model_str += (' ' * max(0, 12 - len(short_name))) + '   ' + ('%s' % short_name) + '|'
			mean_std_str += (' ' * max(0, len(short_name) - 12)) + '   ' + ('%5.2f(%5.2f)' % (mean[-1], std[-1])) + '|'
	if method_type == 'ising':
		plt.ylim([0, 2])
	elif method_type == 'contamination':
		plt.ylim([21.2, 22.5])
	print(model_str)
	print(mean_std_str)
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), fancybox=True, ncol=4)
	plt.xlabel('Number of evaluations')
	plt.tight_layout(rect=[0, 0.00, 1, 0.92])
	plt.show()


if __name__ == '__main__':
	# plotting_mean_std(ising_data(), lamda=0.0000, method_type='ising')
	# plotting_mean_std(ising_data(), lamda=0.0001, method_type='ising')
	# plotting_mean_std(ising_data(), lamda=0.01, method_type='ising')
	plotting_mean_std(contamination_data(), lamda=0.0000, method_type='contamination')
	plotting_mean_std(contamination_data(), lamda=0.0001, method_type='contamination')
	plotting_mean_std(contamination_data(), lamda=0.01, method_type='contamination')
	# plotting_mean_std(aero_data(), lamda=0.01, title_str='Aerostructural Multicomponent')