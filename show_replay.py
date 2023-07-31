from tkinter import *
from time import sleep
from pickle import load
from sys import argv


def path2individual_replay(iter_,n):
	return f'history_buffer/CMA_ES/Bare_minimum/iteration_{iter_}/individual_{n}.pkl'


def path2iteration_info(iter_):
	return f'history_buffer/CMA_ES/Bare_minimum/iteration_{iter_}/iteration_info.pkl'


def parse_arguments(argument_list):
	try:
		arguments = dict(
			(argument_list[i],argument_list[i+1]) for i in range(0, len(argument_list),2)
		)
	except:
		print('possible keys: -m, -iter, -ind')
		print('required format: -key key_value')
		raise ValueError
	return arguments


def display_history_file(history_track):
	root = Tk()
	canvas = Canvas(root, width=250, height=170, bg="gray")
	canvas.pack()
	canvas.focus_set()

	# TILE_TYPES = {"EMPTY"  :0,
	# 				"BODY"   :1,
	# 				"FOOD"   :2,
	# 				"BARRIER":3}
	COLOR_TABLE = ["gray", "orange", "green", "black"]

	for time_counter, score_counter, head_position, game_state in history_track:
		canvas.delete(ALL)
		canvas.create_text(210, 30, text=str(score_counter), font=('Courier',34), fill="green")
		canvas.create_text(210, 70, text=str(time_counter), font=('Courier',34), fill="black")

		for y in range(len(game_state)):
			for x in range(len(game_state[1])):
				canvas.create_rectangle(x*10, y*10, (x+1)*10, (y+1)*10, \
					fill=COLOR_TABLE[game_state[y][x]] \
					if (x,y) not in [head_position] else "red", width=0)

		root.update_idletasks()
		root.update()
		sleep(0.1)


def read_and_display_history_file(path):
	with open(path, 'rb') as f:
		history_track = load(f)
	display_history_file(history_track)


if __name__ == '__main__':
	assert len(argv)!=0,\
		'ERROR: arguments required'
	arguments = parse_arguments(argv[1:])

	if (mode:=arguments.get('-m')):
		if mode=='single-replay':
			if (iter_:=arguments.get('-iter')) and (n:=arguments.get('-ind')):
				path = path2individual_replay(iter_,n)
			else:
				path = 'history_buffer/tmp/replay.pkl'
			read_and_display_history_file(path)

		elif mode=='top-5':
			if (iter_:=arguments.get('-iter')):
				path = path2iteration_info(iter_)
				with open(iteration_info_path, 'rb') as f:
					# population_genome, loss_values, top_score_individuals
					_, _, scores = load(f)
				print(f'top individuals for iter {iter_} are: {scores[:5]}')
				for ind_n in scores[:5]:
					read_and_display_history_file(
						path2individual_replay(iter_, ind_n)
					)
					sleep(0.5)
			else:
				raise ValueError('iteration number should be specified for "top-5" visualization mode')

		elif mode=='top-10':
			pass

		else:
			raise ValueError('possible visualization mode values: {}'.\
								format('single-replay, top-5, top-10'))
	else:
		raise ValueError('visualization mode (-m) should be specified')
