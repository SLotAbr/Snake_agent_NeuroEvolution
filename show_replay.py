from tkinter import *
from time import sleep
from pickle import load
from sys import argv


def path2individual_replay(iter_,n):
	return f'history_buffer/CMA_ES/Bare_minimum/iteration_{iter_}/individual_{n}.pkl'


def load_history(argv):
	mode = ''
	if len(argv)>1:
		if len(argv)==3:
			mode, iter_ = argv[1], argv[2]
			if mode == 'top-5':
				iteration_info_path = f'history_buffer/CMA_ES/Bare_minimum/iteration_{iter_}/iteration_info.pkl'
				with open(iteration_info_path, 'rb') as f:
					_, _, scores = load(f)
				return _, scores
		elif mode == 'single_replay':
			iter_, n = argv[2], argv[3]
			path = path2individual_replay(iter_,n)
	else:
		path = 'history_buffer/tmp/replay.pkl'
	with open(path, 'rb') as f:
		history_track = load(f)
	return history_track, []


def display_history_file(history_track):
	root = Tk()
	canvas = Canvas(root, width=250, height=170, bg="gray")
	canvas.pack()
	canvas.focus_set()

	# TILE_TYPES = {"EMPTY"  :0,
	# 			  "BODY"   :1,
	# 			  "HEAD"   :2,
	# 			  "FOOD"   :3,
	# 			  "BARRIER":4}
	COLOR_TABLE = ["gray", "orange", "red", "green", "black"]

	for time_counter, score_counter, game_state in history_track:
		canvas.delete(ALL)
		canvas.create_text(210, 30, text=str(score_counter), font=('Courier',34), fill="green")
		canvas.create_text(210, 70, text=str(time_counter), font=('Courier',34), fill="black")

		for y in range(len(game_state)):
			for x in range(len(game_state[1])):
				canvas.create_rectangle(x*10, y*10, (x+1)*10, (y+1)*10, \
					fill=COLOR_TABLE[game_state[y][x]], width=0)

		root.update_idletasks()
		root.update()
		sleep(0.1)


history_track, scores = load_history(argv)
if not scores:
	display_history_file(history_track)
else:
	print(f'top individuals for iter {argv[2]} are: {scores[:5]}')
	for ind_n in scores[:5]:
		path = path2individual_replay(argv[2], ind_n)
		with open(path, 'rb') as f:
			history_track = load(f)
		display_history_file(history_track)
		sleep(0.5)
