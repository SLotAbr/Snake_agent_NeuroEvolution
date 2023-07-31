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
		loss, scores = load(f)
	smooth_loss, exponential_window_size = [], 100
	alpha = 1/exponential_window_size
	for e in loss:
		# if abs(e)<500: #outliers handling
		if len(smooth_loss)==0:
			smooth_loss.append(e)
		else:
			# exponential smoothing
			smooth_loss.append( e*alpha+smooth_loss[-1]*(1-alpha) )

	min_value = min(smooth_loss)
	print('minimum value reached at iteration {}: {}'.\
			format(smooth_loss.index(min_value), min_value))
	plt.plot(smooth_loss)
	# plt.plot(scores, label='max scores')
	plt.title(f'smoothed loss values, window_size={exponential_window_size}')
	plt.ylabel('smoothed loss value')
	plt.xlabel('iteration')
	# plt.legend()
	plt.show()
