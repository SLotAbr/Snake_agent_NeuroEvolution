from tkinter import *
from time import sleep
from pickle import load


with open('history_buffer/tmp/replay.pkl', 'rb') as f:
	history_track = load(f)


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
	sleep(0.4)

