import os


BENCHMARK_DATA_DIR = '/home/coh1/Downloads/mse18-incomplete-unweighted-benchmarks'


def problem_size(data_dir=BENCHMARK_DATA_DIR):
	fnames = sorted(os.listdir(data_dir))

	for fname in fnames:
		ffname = os.path.join(data_dir, fname)
		f = open(ffname, 'rt')
		line_str = f.readline()
		while line_str[:2] != 'p ':
			line_str = f.readline()
		nbvar = int(line_str.split(' ')[2])
		nbclause = int(line_str.split(' ')[3])
		hardtag = int(line_str.split(' ')[4])
		if nbvar <= 200:
			f = open(ffname, 'rt')
			line_str = f.readline()
			while line_str[:2] != 'p ':
				line_str = f.readline()
			has_hard_clause = [(int(clause_str.split(' ')[0]) == hardtag) for clause_str in f.readlines()].count(True) > 0
			if not has_hard_clause:
				print('%4d, %6d - %8.2f KB (has hard clauses? : %5s) %s' % (nbvar, nbclause, os.path.getsize(os.path.join(ffname)) / 1024., has_hard_clause, fname))
		f.close()


if __name__ == '__main__':
	problem_size()