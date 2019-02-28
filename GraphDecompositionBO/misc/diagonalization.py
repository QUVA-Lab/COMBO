import os
import pickle
import socket
import math
from datetime import datetime

import numpy as np


def diagonalization_time(n):
	print('Random generation begins at %s ' % datetime.now().strftime("%H:%M:%S:%f"))
	a = np.random.normal(0, 1, [n, n])
	a = np.matmul(a.T, a) + 0.001 * np.eye(n)
	start_time = datetime.now().strftime("%H:%M:%S:%f")
	print('Diagonalization begins   at %s' % start_time)
	eigval, eigvec = np.linalg.eigh(a)
	end_time = datetime.now().strftime("%H:%M:%S:%f")
	print('Diagonalization finished at %s' % end_time)


if __name__ == '__main__':
	diagonalization_time(4900)