from copy import deepcopy
from pickle import dump
from random import randint
from torch import Tensor, zeros
# from sys import argv


def is_position_visibile(coords,field_size):
	return (0<coords[0]<field_size-1) and (0<coords[1]<field_size-1)


def encode_game_state(game_state, 
					  TYPE_dictionary, 
					  field_size, 
					  head_position, 
					  direction):
	reception_range = range(-(field_size-3), field_size-2)
	# (N, c_in, H, W)
	encoding = zeros((1,\
					  len(TYPE_dictionary),\
					  len(reception_range),\
					  len(reception_range)))
	mask = [(head_position[0] + i, head_position[1] + j)
				for i in reception_range
				for j in reception_range]
	for x, y in mask:
		item = TILE_TYPES["BARRIER"] if not is_position_visibile((x,y),field_size) \
										else game_state[y][x]
		encoding[0][item][y][x] = 1
	return encoding


def save_replay(history_track, path='history_buffer/tmp/replay.pkl'):
	# with open('history_buffer/{agent\'s name}/replay.pkl','wb') as f:
	with open(path,'wb') as f:
		dump(history_track, f)


def run_simulation():
	FIELD_SIZE = 9
	# agent_path = argv[1]
	is_crashed, time_counter, score_counter = False, 0, 0
	history_track, snake_postions = [], [(3,2),(2,2)] 
	food_position = tuple(randint(1, FIELD_SIZE-2) for _ in range(2))
	# "UP":0, "RIGHT":1, "DOWN":2, "LEFT":3
	direction_id = 1
	DIRECTION = [(0,1), (1,0), (0,-1), (-1,0)]
	TILE_TYPES = {"EMPTY"  :0,
				  "BODY"   :1,
				  "HEAD"   :2,
				  "FOOD"   :3,
				  "BARRIER":4}
	game_state = []
	coord_comparsion = lambda a,b: a[0]==b[0] and a[1]==b[1]
	for y in range(FIELD_SIZE):
		y_line = []
		for x in range(FIELD_SIZE):
			if (x == 0) or (x == FIELD_SIZE-1) or \
			   (y == 0) or (y == FIELD_SIZE-1):
				y_line.append(TILE_TYPES["BARRIER"])
			elif (x,y) in snake_postions:
				y_line.append(TILE_TYPES["BODY"])
			elif coord_comparsion((x,y), food_position):
				y_line.append(TILE_TYPES["FOOD"])
			else:
				y_line.append(TILE_TYPES["EMPTY"])
		game_state.append(y_line)
	game_state[snake_postions[0][1]][snake_postions[0][0]] = TILE_TYPES["HEAD"]
	history_track.append((time_counter, score_counter, deepcopy(game_state)))

	while not is_crashed:
		# encode game_state and send it to the agent
		# reception_field = encode_game_state(game_state,\
		# 									TILE_TYPES,\
		# 									FIELD_SIZE,\
		# 									snake_postions[0],\
		# 									-1)
		# action = agent_API(reception_field)
		# could be: -1,0,1
		action = 1 if randint(0,8)>5 else 0
		direction_id = (direction_id+action)%4

		# agent's action time
		new_x, new_y = snake_postions[0]
		new_x += DIRECTION[direction_id][0]
		new_y += DIRECTION[direction_id][1]

		if coord_comparsion((new_x,new_y), food_position):
			score_counter+=1
			game_state[snake_postions[0][1]][snake_postions[0][0]] = TILE_TYPES["BODY"]
			while True:
				food_position = tuple(randint(1, FIELD_SIZE-2) for _ in range(2))
				if food_position not in snake_postions:
					game_state[food_position[1]][food_position[0]] = TILE_TYPES["FOOD"]
					break
		elif game_state[new_y][new_x]==TILE_TYPES["BARRIER"] or\
			  (new_x,new_y) in snake_postions:
			game_state[snake_postions[-1][1]][snake_postions[-1][0]] = TILE_TYPES["EMPTY"]
			game_state[snake_postions[0][1]][snake_postions[0][0]] = TILE_TYPES["BODY"]
			is_crashed = True
		else:
			game_state[snake_postions[-1][1]][snake_postions[-1][0]] = TILE_TYPES["EMPTY"]
			game_state[snake_postions[0][1]][snake_postions[0][0]] = TILE_TYPES["BODY"]
			snake_postions.pop()

		game_state[new_y][new_x] = TILE_TYPES["HEAD"]
		snake_postions.insert(0, [new_x,new_y])

		time_counter += 1
		history_track.append((time_counter, score_counter, deepcopy(game_state)))

	save_replay(history_track)
