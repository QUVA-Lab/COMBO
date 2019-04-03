import os
import socket
import pickle
import time
from datetime import datetime

import torch


def experiment_directory():
	hostname = socket.gethostname()
	if hostname == 'DTA160000':
		return '/home/coh1/Experiments/GraphDecompositionBO'
	elif hostname[:4] == 'node':
		return '/var/scratch/coh/Experiments/GraphDecompositionBO'
	else:
		raise ValueError('Set proper experiment directory on your machine.')


def model_data_filenames(exp_dir, objective_name):
	folder_name = objective_name + '_' + datetime.now().strftime('%Y%m%d-%H:%M:%S:%f')
	os.makedirs(os.path.join(exp_dir, folder_name))
	logfile_dir = os.path.join(exp_dir, folder_name, 'log')
	os.makedirs(logfile_dir)
	model_filename = os.path.join(exp_dir, folder_name, 'model.pt')
	cfg_data_filename = os.path.join(exp_dir, folder_name, 'data_config.pkl')
	return model_filename, cfg_data_filename, logfile_dir


def load_model_data(path, exp_dir=experiment_directory()):
	if not os.path.exists(path):
		path = os.path.join(exp_dir, path)
	logfile_dir = os.path.join(path, 'log')
	model_filename = os.path.join(path, 'model.pt')
	cfg_data_filename = os.path.join(path, 'data_config.pkl')

	model = torch.load(model_filename)
	cfg_data_file = open(cfg_data_filename, 'r')
	cfg_data = pickle.load(cfg_data_file)
	for key, value in pickle.load(cfg_data_file).iteritems():
		if key != 'logfile_dir':
			exec (key + '=value')
	cfg_data_file.close()
	
	return model, cfg_data, logfile_dir


def save_model_data(model, model_filename, cfg_data, cfg_data_filename):
	torch.save(model, model_filename)
	f = open(cfg_data_filename, 'w')
	pickle.dump(cfg_data, f)
	f.close()


def displaying_and_logging(logfile_dir, eval_inputs, eval_outputs, pred_mean_list, pred_std_list, pred_var_list, time_list, elapse_list):
	logfile = open(os.path.join(logfile_dir, str(eval_inputs.size(0)).zfill(4) + '.out'), 'w')
	for i in range(eval_inputs.size(0)):
		min_val, min_ind = torch.min(eval_outputs[:i + 1], 0)
		time_str = time.strftime('%H:%M:%S', time.gmtime(time_list[i])) + '(' + time.strftime('%H:%M:%S', time.gmtime(elapse_list[i])) + ')  '
		data_str = ('%3d-th : %+12.4f, '
		            'mean : %+.4E, '
		            'std : %.4E, '
		            'var : %.4E, ' 
		            'min : %+8.4f(%3d)' %
		            (i + 1, eval_outputs.squeeze()[i],
		             pred_mean_list[i],
		             pred_std_list[i],
		             pred_var_list[i],
		             min_val.item(), min_ind.item() + 1))
		min_str = '  <==== IMPROVED' if i == min_ind.data.item() else ''
		print(time_str + data_str + min_str)
		logfile.writelines(time_str + data_str + min_str + '\n')
	logfile.close()