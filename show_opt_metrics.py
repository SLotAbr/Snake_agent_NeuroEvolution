import matplotlib.pyplot as plt
from pickle import load
from sys import argv


def get_path(optimizer_id='CMA_ES', agent_id='Bare_minimum'):
	return f'history_buffer/{optimizer_id}/{agent_id}/optimizer_metrics.pkl'


def parse_arguments(argument_list):
	try:
		arguments = dict()
		if not argument_list:
			return arguments
		argument_list = argument_list[::-1]
		while argument_list:
			item = argument_list.pop()
			if item in ('-h', '-e', '-s'):
				arguments.update([(item, -1)])
			else:
				value = argument_list.pop()
				arguments.update([(item, value)])
	except:
		print('possible keys: -opt, -agent, -h, -e, -w, -s')
		print('values for the following keys should be specified:', end='')
		print('-opt, -agent, -w')
		raise ValueError
	return arguments


if __name__ == '__main__':
	"""
	-h : handle outliers
	-e : exponential smoothing
	-w : exponential_window_size
	-s : scores values instead of loss values
	"""
	arguments = parse_arguments(argv[1:])
	opt_id = opt if (opt:=arguments.get('-opt')) else 'CMA_ES'
	agent_id = agent if (agent:=arguments.get('-agent')) else 'Bare_minimum'
	with open(get_path(opt_id, agent_id), 'rb') as f:
		# it's median loss values for iterations
		loss, scores = load(f)

	if not arguments.get('-s'):
		if arguments.get('-e'):
			smooth_loss = []
			exponential_window_size = int(window)\
				if (window:=arguments.get('-w')) else 100
			alpha = 1/exponential_window_size
			for e in loss:
				if arguments.get('-h'):
					if abs(e)>=500:
						continue

				if len(smooth_loss)==0:
					smooth_loss.append(e)
				else:
					smooth_loss.append( e*alpha+smooth_loss[-1]*(1-alpha) )
			loss = smooth_loss

		elif arguments.get('-h'):
			loss = [e for e in loss if abs(e)<500]

		min_value = min(loss)
		print('minimum loss value reached at iteration {}: {}'.\
				format(loss.index(min_value), min_value))
		plt.plot(loss)

		if arguments.get('-e'):
			plt.title(f'smoothed loss values, window_size={exponential_window_size}')
			plt.ylabel('smoothed loss value')
		else:
			plt.title(f'loss values')
			plt.ylabel('loss value')
	else:
		# plt.plot(scores, label='max scores')
		max_scores = max(scores)
		print('max scores reached at iteration {}: {}'.\
				format(scores.index(max_scores), max_scores))
		plt.plot(scores)

		plt.title(f'max scores')
		plt.ylabel('max score')

	plt.xlabel('iteration')
	# plt.legend()
	plt.show()
