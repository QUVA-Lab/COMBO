import socket


def result_dir():
	hostname = socket.gethostname()
	if hostname == 'DTA160000':
		return '/home/coh1/Experiments/CombinatorialBO'
	elif hostname in ['u031490', 'quva-peter', 'U036713']:
		return '/home/changyongoh/Experiments/CombinatorialBO'
	else:
		raise NotImplementedError


def exp_dir():
	hostname = socket.gethostname()
	if hostname == 'DTA160000':
		return '/home/coh1/Experiments/CombinatorialBO_SMAC'
	elif hostname in ['u031490', 'quva-peter', 'U036713']:
		return '/home/changyongoh/Experiments/CombinatorialBO_SMAC'
	else:
		raise NotImplementedError