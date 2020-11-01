import os
import pickle
import time
from datetime import datetime

import torch
from COMBO.config import experiment_directory


def bo_exp_dirname(exp_dir, objective_name):
	folder_name = objective_name + '_' + datetime.now().strftime('%Y%m%d-%H:%M:%S:%f')
	os.makedirs(os.path.join(exp_dir, folder_name))
	logfile_dir = os.path.join(exp_dir, folder_name, 'log')
	os.makedirs(logfile_dir)
	return os.path.join(exp_dir, folder_name)


def displaying_and_logging(logfile_dir, eval_inputs, eval_outputs, pred_mean_list, pred_std_list, pred_var_list,
                           time_list, elapse_list, store_data=False):
	logfile = open(os.path.join(logfile_dir, str(eval_inputs.size(0)).zfill(4) + '.out'), 'w')
	for i in range(eval_inputs.size(0)):
		min_val, min_ind = torch.min(eval_outputs[:i + 1], 0)
		time_str = time.strftime('%H:%M:%S', time.gmtime(time_list[i])) \
		           + '(' + time.strftime('%H:%M:%S', time.gmtime(elapse_list[i])) + ')  '
		data_str = ('%3d-th : %+12.4f, mean : %+.4E, std : %.4E, var : %.4E, min : %+8.4f(%3d)' %
		            (i + 1, eval_outputs.squeeze()[i],
		             pred_mean_list[i], pred_std_list[i], pred_var_list[i],
		             min_val.item(), min_ind.item() + 1))
		min_str = '  <==== IMPROVED' if i == min_ind.item() else ''
		print(time_str + data_str + min_str)
		logfile.writelines(time_str + data_str + min_str + '\n')
	logfile.close()
	if store_data:
		pklfilename = os.path.join(logfile_dir, str(eval_inputs.size(0)).zfill(4) + '.pkl')
		torch.save({'inputs': eval_inputs, 'outputs': eval_outputs}, pklfilename)
