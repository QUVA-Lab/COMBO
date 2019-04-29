import socket


def experiment_directory():
	hostname = socket.gethostname()
	if hostname == 'DTA160000':
		return '/home/coh1/Experiments/GraphDecompositionBO'
	elif hostname[:4] == 'node':
		return '/var/scratch/coh/Experiments/GraphDecompositionBO'
	else:
		raise ValueError('Set proper experiment directory on your machine.')


def data_directory():
	hostname = socket.gethostname()
	if hostname == 'DTA160000':
		return '/home/coh1/DATA'
	elif hostname[:4] == 'node':
		return '/var/scratch/coh/DATA'
	else:
		raise ValueError('Set proper experiment directory on your machine.')