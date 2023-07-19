import matplotlib.pyplot as plt
from pickle import load
from sys import argv


def get_path(optimizer_id='CMA_ES', agent_id='Bare_minimum'):
	return f'history_buffer/{optimizer_id}/{agent_id}/optimizer_metrics.pkl'


if __name__ == '__main__':
	# assert len(argv)==1,\
	# 	'optimizer id and agent id should be specified'
	opt_id, agent_id = ('CMA_ES', 'Bare_minimum') \
							if len(argv)==1 else (argv[1], argv[2])
	with open(get_path(opt_id, agent_id), 'rb') as f:
		fitness, scores = load(f)
	plt.plot(fitness)
	# plt.plot(scores, label='max scores')
	plt.ylabel('fitness value')
	plt.xlabel('iteration')
	# plt.legend()
	plt.show()
