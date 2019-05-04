import socket


PROGRESS_BAR_LEN = 50


def experiment_directory():
	hostname = socket.gethostname()
	if hostname == 'DTA160000':
		return '/home/coh1/Experiments/GraphDecompositionBO'
	elif hostname == 'quva01':
		return '/home/changyongoh/Experiments/GraphDecompositionBO'
	elif hostname[-16:] == 'lisa.surfsara.nl':
		return '/home/cyoh/Experiments/GraphDecompositionBO'
	elif hostname[:4] == 'node':
		return '/var/scratch/coh/Experiments/GraphDecompositionBO'
	else:
		raise ValueError('Set proper experiment directory on your machine.')


def data_directory():
	hostname = socket.gethostname()
	if hostname == 'DTA160000':
		return '/home/coh1/Data'
	elif hostname == 'quva01':
		return '/home/changyongoh/Data'
	elif hostname[-16:] == 'lisa.surfsara.nl':
		return '/home/cyoh/Data'
	elif hostname[:4] == 'node':
		return '/var/scratch/coh/Data'
	else:
		raise ValueError('Set proper experiment directory on your machine.')


def SMAC_exp_dir():
	hostname = socket.gethostname()
	if hostname == 'DTA160000' or hostname[:6] == 'ivi-cn':
		return '/home/coh1/Experiments/CombinatorialBO_SMAC'
	elif hostname == 'quva01':
		return '/home/changyongoh/Experiments/CombinatorialBO_SMAC'
	elif hostname[-16:] == 'lisa.surfsara.nl':
		return '/home/cyoh/Experiments/CombinatorialBO_SMAC'
	elif hostname[:4] == 'node':
		return '/var/scratch/coh/Experiments/CombinatorialBO_SMAC'
	else:
		raise NotImplementedError