import socket


def result_dir():
	hostname = socket.gethostname()
	if hostname == 'DTA160000':
		return '/home/coh1/Experiments/GraphDecompositionBO'
	elif hostname in ['u031490', 'quva-peter', 'U036713']:
		return '/home/changyongoh/Experiments/GraphDecompositionBO'
	else:
		raise NotImplementedError


def exp_dir():
	hostname = socket.gethostname()
	if hostname == 'DTA160000':
		return '/home/coh1/Experiments/GraphDecompositionBO_SMAC'
	elif hostname in ['u031490', 'quva-peter', 'U036713']:
		return '/home/changyongoh/Experiments/GraphDecompositionBO_SMAC'
	else:
		raise NotImplementedError