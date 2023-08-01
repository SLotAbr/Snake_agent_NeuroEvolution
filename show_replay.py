import numpy as np
from os import listdir
from pickle import load
from sys import argv
from time import sleep
from tkinter import *


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


def update_agent_info(iteration_number, min_values, agent_info):
	try:
		with open(path2iteration_info(iteration_number), 'rb') as f:
			_, fitness_list, _ = load(f)
		sort_indexes = np.argsort(fitness_list)
		for i in sort_indexes:
			candidate = max(min_values)
			if fitness_list[i] < candidate:
				x = min_values.index(candidate)
				min_values[x] = fitness_list[i]
				agent_info[x] = (iteration_number, i)
			else:
				break
	except:
		print(f'ERROR during iteration_{iteration_number} folder reading')
		print("iteration_info.pkl wasn't found")
		print('This directory will be ignored')
	return min_values, agent_info


def display_history_file(history_track):
	root = Tk()
	canvas = Canvas(root, width=170, height=90, bg="gray")
	canvas.pack()
	canvas.focus_set()

	# TILE_TYPES = {"EMPTY"  :0,
	# 				"BODY"   :1,
	# 				"FOOD"   :2,
	# 				"BARRIER":3}
	COLOR_TABLE = ["gray", "orange", "green", "black"]

	for time_counter, score_counter, head_position, game_state in history_track:
		canvas.delete(ALL)
		canvas.create_text(130, 30, text=str(score_counter), font=('Courier',34), fill="green")
		canvas.create_text(130, 70, text=str(time_counter), font=('Courier',34), fill="black")

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


def display_several_history_files(replays_info, delay=0.5):
	"replays_info item: (iteration_number, individual_number)"
	for iter_, ind_n in replays_info:
		read_and_display_history_file(
			path2individual_replay(iter_, ind_n)
		)
		sleep(delay)


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
				with open(path, 'rb') as f:
					# population_genome, loss_values, top_score_individuals
					_, _, scores = load(f)
				print(f'top-5 individuals for iter {iter_} are: {scores[:5]}')
				display_several_history_files([(iter_, e) for e in scores[:5]])
			else:
				raise ValueError('iteration number should be specified for "top-5" visualization mode')

		elif mode=='top-10':
			agent_folder = listdir('history_buffer/CMA_ES/Bare_minimum')
			if len(agent_folder)!=0:
				min_values = [99999 for _ in range(10)]
				agent_info =[(-1,-1) for _ in range(10)]

				for folder_name in agent_folder:
					if 'iteration' in folder_name:
						iteration_number = folder_name.split('_')[1]
						min_values, agent_info = update_agent_info(
								iteration_number, min_values, agent_info
						)

				print('top-10 individuals for the entire optimization are:')
				order = np.argsort(min_values)
				for i in order:
					print('iteration {}, individual {} with loss value: {}'.\
							format(agent_info[i][0], agent_info[i][1], min_values[i])
					)
				display_several_history_files(np.array(agent_info)[order])
			else:
				raise SystemError('iteration_info not found: start the optimization first')

		else:
			raise ValueError('possible visualization mode values: {}'.\
								format('single-replay, top-5, top-10'))
	else:
		raise ValueError('visualization mode (-m) should be specified')
