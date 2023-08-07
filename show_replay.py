import numpy as np
from os import listdir, remove
from pickle import load
from sys import argv
from time import sleep
from tkinter import *


def path2individual_replay(iter_, n, agent_id='Bare_minimum'):
	return f'history_buffer/CMA_ES/{agent_id}/iteration_{iter_}/individual_{n}.pkl'


def path2iteration_info(iter_, agent_id='Bare_minimum'):
	return f'history_buffer/CMA_ES/{agent_id}/iteration_{iter_}/iteration_info.pkl'


def parse_arguments(argument_list):
	try:
		arguments = dict(
			(argument_list[i],argument_list[i+1]) for i in range(0, len(argument_list),2)
		)
	except:
		print('possible keys: -m, -iter, -ind, -s')
		print('required format: -key key_value')
		raise ValueError
	return arguments


def update_agent_info(iteration_number, min_values, agent_info, agent_id):
	try:
		with open(path2iteration_info(iteration_number, agent_id), 'rb') as f:
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


def image_grab(coords, game_step):
	# the first grab is too fast and catches a screen area under the widget
	if game_step==0:
		sleep(1)
	img = ImageGrab.grab(coords)
	img.save(f'history_buffer/tmp/game_state_{game_step}.png')


def display_history_file(history_track, save_mode=False):
	root = Tk()
	canvas = Canvas(root, width=170, height=90, bg="gray")
	canvas.pack()
	canvas.focus_set()

	# TILE_TYPES = {"EMPTY"  :0,
	# 				"BODY"   :1,
	# 				"FOOD"   :2,
	# 				"BARRIER":3}
	COLOR_TABLE = ["gray", "orange", "green", "black"]

	for i in range(len(history_track)):
		time_counter, score_counter, head_position, game_state = history_track[i]
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
		if save_mode:
			x0 = canvas.winfo_rootx()
			y0 = canvas.winfo_rooty()
			x1 = x0 + canvas.winfo_width()
			y1 = y0 + canvas.winfo_height()
			image_grab((x0, y0, x1, y1), i)
		sleep(0.01)


def read_and_display_history_file(path, save_mode=''):
	with open(path, 'rb') as f:
		history_track = load(f)
	display_history_file(
		history_track, save_mode=True if save_mode=='GIF' else False
	)


def display_several_history_files(replays_info, delay=0.5):
	"replays_info item: (iteration_number, individual_number)"
	for iter_, ind_n, agent_id in replays_info:
		read_and_display_history_file(
			path2individual_replay(iter_, ind_n, agent_id)
		)
		sleep(delay)


def create_gif_replay(iter_, 
					  n, 
					  agent_id='Bare_minimum', 
					  frame_folder='history_buffer/tmp'):
	frame_paths = glob(f"{frame_folder}/*.png")
	frames = [Image.open(image) for image in sorted(
		frame_paths,
		key=lambda s: int(findall(r'\d+', s)[0])
	)]
	# a pause for the beginning and the end of an animation
	frames = [frames[0] for _ in range(5)] + frames
	frames.extend([frames[-1] for _ in range(5)])
	frame_one = frames[0]
	frame_one.save(
		f"GIFs/CMA_ES-{agent_id}-iteration_{iter_}-individual_{n}.gif",
		format="GIF", append_images=frames, save_all=True, duration=100, loop=0
	)
	[remove(e) for e in frame_paths]


if __name__ == '__main__':
	assert len(argv)!=0,\
		'ERROR: arguments required'
	arguments = parse_arguments(argv[1:])
	agent_id = agent if (agent:=arguments.get('-agent')) else 'Bare_minimum'

	if (mode:=arguments.get('-m')):
		if mode=='single-replay':
			if (iter_:=arguments.get('-iter')) and (n:=arguments.get('-ind')):
				path = path2individual_replay(iter_, n, agent_id)
			else:
				path = 'history_buffer/tmp/replay.pkl'

			if (save_mode:=arguments.get('-s')) and iter_ and n:
				from glob import glob
				from re import findall
				from PIL import Image, ImageGrab
				read_and_display_history_file(path, save_mode=save_mode)
				create_gif_replay(iter_,n, agent_id)
			else:
				read_and_display_history_file(path)

		elif mode=='top-5':
			if (iter_:=arguments.get('-iter')):
				path = path2iteration_info(iter_, agent_id)
				with open(path, 'rb') as f:
					# population_genome, loss_values, top_score_individuals
					_, _, scores = load(f)
				print(f'top-5 {agent_id} individuals for iter {iter_} are: {scores[:5]}')
				display_several_history_files([(iter_, e, agent_id) for e in scores[:5]])
			else:
				raise ValueError('iteration number should be specified for "top-5" visualization mode')

		elif mode=='top-10':
			agent_folder = listdir(f'history_buffer/CMA_ES/{agent_id}')
			if len(agent_folder)!=0:
				min_values = [99999 for _ in range(10)]
				agent_info =[(-1,-1) for _ in range(10)]

				for folder_name in agent_folder:
					if 'iteration' in folder_name:
						iteration_number = folder_name.split('_')[1]
						min_values, agent_info = update_agent_info(
								iteration_number, min_values, agent_info, agent_id
						)

				print(f'top-10 {agent_id} individuals for the entire optimization are:')
				order = np.argsort(min_values)
				for i in order:
					print('iteration {}, individual {} with loss value: {}'.\
							format(agent_info[i][0], agent_info[i][1], min_values[i])
					)
				display_several_history_files(
					[(e_0, e_1, agent_id) for e_0, e_1 in np.array(agent_info)[order]]
				)
			else:
				raise SystemError('iteration_info not found: start the optimization first')

		else:
			raise ValueError('possible visualization mode values: {}'.\
								format('single-replay, top-5, top-10'))
	else:
		raise ValueError('visualization mode (-m) should be specified')
