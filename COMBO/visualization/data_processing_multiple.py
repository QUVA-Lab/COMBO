import os
import pickle
import cPickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


PEST_COMBO_DATA_DIR = '/home/coh1/Experiments/COMBO/PestControl_COMBO'
CENTROID_COMBO_DATA_DIR = '/home/coh1/Experiments/COMBO/Centroid_COMBO'
BRANIN_COMBO_DATA_DIR = '/home/coh1/Experiments/COMBO/Branin_COMBO'


def name_in_plot(name):
	if 'COMBO' in name:
		return name
	elif name == 'randomsearch':
		return 'RS'
	elif name == 'smac':
		return 'SMAC'
	elif name == 'simulatedannealing':
		return 'SA'
	elif name == 'tpe':
		return 'TPE'
	else:
		raise ValueError('Not supported algorithm name')


def assign_color(model_name):
	if model_name == 'COMBO':
		return 'k'
	elif model_name == 'smac':
		return 'm'
	elif model_name == 'simulatedannealing':
		return 'g'
	elif model_name == 'randomsearch':
		return 'r'
	elif model_name == 'tpe':
		return 'c'
	return np.random.RandomState(sum([ord(elm) + 1 for elm in model_name]) + 2).rand(3)


def baseline_data(identifier, data_dir='/home/coh1/Experiments/COMBO'):
	mean_std_dict = {}
	for elm in os.listdir(data_dir):
		if identifier == elm.split('_')[0]:
			algo_name = os.path.split(elm)[1][:-4].split('_')[-1]
			data_file = open(os.path.join(data_dir, elm))
			data = pickle.load(data_file)
			data_file.close()
			print(algo_name)
			mean_std_dict[algo_name] = data
	return mean_std_dict


def collection_combo_data(data_dir, identifier):
	evaluations = None
	for elm in os.listdir(data_dir):
		if identifier in elm:
			evaluation_sample = directory_python_data(os.path.join(data_dir, elm))
			if evaluations is None:
				evaluations = evaluation_sample
			elif evaluations.shape[0] > evaluation_sample.shape[0]:
				print('%d vs %d when stacking %s' % (evaluations.shape[0], evaluation_sample.shape[0], elm))
				evaluations = np.hstack([evaluations, np.vstack((evaluation_sample, np.zeros((evaluations.shape[0] - evaluation_sample.shape[0], 1))))])
			elif evaluation_sample.shape[0] > evaluations.shape[0]:
				print('%d vs %d when stacking %s' % (evaluations.shape[0], evaluation_sample.shape[0], elm))
				evaluations = np.hstack([np.vstack((evaluations, np.zeros((evaluation_sample.shape[0] - evaluations.shape[0], evaluations.shape[1])))), evaluations])
			else:
				evaluations = np.hstack([evaluations, evaluation_sample])
	mean_std_dict = {'COMBO': {'mean': np.mean(evaluations, axis=1), 'std': np.std(evaluations, axis=1)}}
	return mean_std_dict


def directory_python_data(data_dir):
	last_log_filename = str(max([int(elm[:4]) for elm in os.listdir(os.path.join(data_dir, 'log'))])).zfill(4) + '.out'
	last_log_file = open(os.path.join(data_dir, 'log', last_log_filename))
	log_lines = last_log_file.readlines()
	last_log_file.close()
	return np.array([objective_from_log_line(log_lines[i]) for i in range(len(log_lines))]).reshape(-1, 1)


def objective_from_log_line(log_line):
	return float(log_line.split()[15].split('(')[0])


def pest_data():
	mean_std_dict = collection_combo_data(PEST_COMBO_DATA_DIR, identifier='PestControl')
	mean_std_dict.update(baseline_data(identifier='pestcontrol'))
	return mean_std_dict


def centroid_data():
	mean_std_dict = collection_combo_data(CENTROID_COMBO_DATA_DIR, identifier='Centroid')
	mean_std_dict.update(baseline_data(identifier='centroid'))
	return mean_std_dict


def branin_data():
	mean_std_dict = collection_combo_data(BRANIN_COMBO_DATA_DIR, identifier='Branin')
	mean_std_dict.update(baseline_data(identifier='branin'))
	return mean_std_dict


def plotting_mean_std(mean_std_dict, lamda, title_str=''):
	assert lamda in [0, 0.0001, 0.01]
	begin_ind = 20
	z_value = 0.5
	model_str = ''
	mean_std_str = ''
	algorithm_list = ['COMBO', 'ExpectedImprovement', 'HorseShoeorder2SDP', 'ObliviousLocalSearch', 'RandomSearch', 'SMAC', 'SequentialMonteCarlo', 'SimulatedAnnealing']
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
	print(model_str)
	print(mean_std_str)
	plt.legend()
	if 'Ising' in title_str:
		plt.ylim([-0.5, 2.0])
	elif 'Contamination' in title_str:
		plt.ylim([21.0, 22.5])
	elif 'Aerostructural' in title_str:
		plt.ylim([0.05, 0.2])
	plt.title(('%s ' + u"\u03BB" +':%.0E') % (title_str, lamda))
	plt.xlabel('Number of Evaluations')
	plt.show()


if __name__ == '__main__':
	# f = open('/home/coh1/Experiments/COMBO/pestcontrol_baseline_result_simulatedannealing.pkl')
	# data = pickle.load(f)
	# f.close()
	# print(data)
	test_case = 'pestcontrol'
	if test_case == 'pestcontrol':
		mean_std_dict = pest_data()
		maximum = min([elm['mean'][0] for elm in mean_std_dict.values()])

		for key, value in mean_std_dict.iteritems():
			x = range(21, 271)
			mean = np.minimum(value['mean'], maximum)
			std = value['std']
			stderr = 25 ** 0.5
			if key in ['randomsearch', 'COMBO']:
				begin_ind = 20
				end_ind = 270
			elif key in ['smac', 'tpe']:
				begin_ind = 0
				end_ind = 250
			elif key in ['simulatedannealing']:
				begin_ind = 1
				end_ind = 251
			color = assign_color(key)
			print(key, mean.shape)
			print(mean[-1])
			print(std[-1] / stderr)
			print('%4.2f\pm%4.2f' % (mean[-1], std[-1] / stderr))
			plot_name = name_in_plot(key)
			plt.plot(x, mean[begin_ind:end_ind], label=plot_name, color=color)
			plt.fill_between(x, mean[begin_ind:end_ind] - std[begin_ind:end_ind] / stderr, mean[begin_ind:end_ind] + std[begin_ind:end_ind] / stderr, alpha=0.25, color=color)
		plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, ncol=5)
		plt.xlabel('Number of evalutions')
		plt.tight_layout(rect=[0, 0.00, 1, 0.95])
		plt.show()
	elif test_case == 'branin':
		mean_std_dict = branin_data()
		maximum = min([elm['mean'][0] for elm in mean_std_dict.values()])

		for key, value in mean_std_dict.iteritems():
			x = range(1, 101)
			mean = np.minimum(value['mean'], maximum)
			std = value['std']
			stderr = 25 ** 0.5
			if key in ['randomsearch', 'COMBO']:
				begin_ind = 0
				end_ind = 100
			elif key in ['smac', 'tpe']:
				begin_ind = 0
				end_ind = 100
			elif key in ['simulatedannealing']:
				begin_ind = 1
				end_ind = 101
			color = assign_color(key)
			plot_name = name_in_plot(key)
			plt.plot(x, mean[begin_ind:end_ind], label=plot_name, color=color)
			plt.fill_between(x, mean[begin_ind:end_ind] - std[begin_ind:end_ind] / stderr,
			                 mean[begin_ind:end_ind] + std[begin_ind:end_ind] / stderr, alpha=0.25, color=color)
			print(key, mean.shape)
			print(mean[-1])
			print(std[-1] / stderr)
			print('%4.2f\pm%4.2f' % (mean[-1], std[-1] / stderr))
		plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, ncol=5)
		plt.xlabel('Number of evalutions')
		plt.ylim([0, 5])
		plt.tight_layout(rect=[0, 0.00, 1, 0.95])
		plt.show()
