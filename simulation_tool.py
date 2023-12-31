import os
# from agents import Rule_based_agent
from copy import deepcopy
from pickle import dump
from random import randint
from torch import Tensor, cat, rot90, zeros
# from sys import argv


def is_position_visibile(coords,field_size):
	return (0<coords[0]<field_size-1) and (0<coords[1]<field_size-1)


def encode_game_state(game_state, 
					  TYPE_dictionary, 
					  field_size, 
					  head_position, 
					  direction):
	reception_range = range(-(field_size-3), field_size-2)
	rr_len = len(reception_range)
	# (N, c_in, H, W)
	encoding = zeros((1,\
					  len(TYPE_dictionary),\
					  rr_len,\
					  rr_len))
	mask = [(head_position[0] + j, head_position[1] + i)
				for i in reception_range
				for j in reception_range]
	for i in range(len(mask)):
		item = TYPE_dictionary["BARRIER"] \
					if not is_position_visibile(mask[i],field_size) \
						else game_state[mask[i][1]][mask[i][0]] # [y][x]
		encoding[0][item][i//rr_len][i%rr_len] = 1
		
	if not (direction==0):
		encoding = rot90(encoding, direction, dims=[2,3])
	return encoding


def generate_food_tile(snake_postions, FIELD_SIZE):
	if len(snake_postions) == (FIELD_SIZE-2)**2:
		return (0,0)
	else:
		while True:
			food_position = tuple(randint(1, FIELD_SIZE-2) for _ in range(2))
			if food_position not in snake_postions:
				return food_position	


def save_replay(history_track, opt_path, progress_info):
	iter_folder_path = opt_path+progress_info.split('/')[0]
	if not os.path.exists(iter_folder_path): os.mkdir(iter_folder_path)
	# with open('history_buffer/{agent\'s name}/replay.pkl','wb') as f:
	with open(opt_path+progress_info,'wb') as f:
		dump(history_track, f)


# def save_VAE_train_data(history_track, mask_id):
# 	with open(f'history_buffer/VAE_data/mask_{mask_id}_replay.pkl','wb') as f:
# 		dump(history_track, f)


def get_unique_game_state_number(history_track, field_size):
	game_state_list = cat(
		[Tensor(game_state).view(1,field_size,field_size) \
			for _, _, _, game_state in history_track]
	)
	return game_state_list.unique(dim=0).shape[0]


def run_simulation(agent_info, opt_info, agent):
	FIELD_SIZE = 9
	is_crashed, time_counter, score_counter, score_list = False, 0, 0, [0]
	history_track, snake_postions = [], [(3,2),(2,2)] 
	food_position = generate_food_tile(snake_postions, FIELD_SIZE)
	last_food_iter = 0 # the last food consumed iteration
	CONSUMPTION_RATE = (FIELD_SIZE-2)**2
	# "UP":0, "RIGHT":1, "DOWN":2, "LEFT":3
	direction_id = 1
	DIRECTION = [(0,-1), (1,0), (0,1), (-1,0)]
	TILE_TYPES = {"EMPTY"  :0,
				  "BODY"   :1,
				  "FOOD"   :2,
				  "BARRIER":3}
	game_state = []; action_list = []
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
	history_track.append(
		(time_counter, score_counter, snake_postions[0], deepcopy(game_state))
	)

	while not is_crashed:
		reception_field = encode_game_state(
			game_state, TILE_TYPES, FIELD_SIZE, snake_postions[0], direction_id
		)
		# could be: -1,0,1
		action = agent(reception_field)
		action_list.append(int(action))
		# action = 1 if randint(0,8)>5 else 0
		direction_id = (direction_id+action)%4

		# agent's action time
		new_x, new_y = snake_postions[0]
		new_x += DIRECTION[direction_id][0]
		new_y += DIRECTION[direction_id][1]

		if coord_comparsion((new_x,new_y), food_position):
			score_counter+=1
			score_list.append(1)
			last_food_iter = time_counter+1
			food_position = generate_food_tile(
				[(new_x,new_y)]+snake_postions, FIELD_SIZE
			)
			if food_position[0]:
				game_state[food_position[1]][food_position[0]] = TILE_TYPES["FOOD"]
		elif game_state[new_y][new_x]==TILE_TYPES["BARRIER"] or\
			  (new_x,new_y) in snake_postions:
			score_list.append(0)
			game_state[snake_postions[-1][1]][snake_postions[-1][0]] = TILE_TYPES["EMPTY"]
			is_crashed = True
		else:
			score_list.append(0)
			game_state[snake_postions[-1][1]][snake_postions[-1][0]] = TILE_TYPES["EMPTY"]
			snake_postions.pop()

		game_state[new_y][new_x] = TILE_TYPES["BODY"]
		snake_postions.insert(0, (new_x,new_y))

		time_counter += 1
		history_track.append(
			(time_counter, score_counter, snake_postions[0], deepcopy(game_state))
		)

		if (time_counter - last_food_iter)>CONSUMPTION_RATE:
			score_list[0] = -200
			break
		if (len(action_list) >= 7) and (len(set(action_list[-7:]))==1):
			score_list[0] = -200
			break

	opt_id, agent_id, iter_ = opt_info[0], agent_info[0], opt_info[1]
	save_replay(history_track,
		f'history_buffer/{opt_id}/{agent_id}/',
		f'iteration_{iter_}/individual_{str(agent_info[1])}.pkl')
	
	exploration_score = get_unique_game_state_number(history_track, FIELD_SIZE)
	return score_list, score_counter, exploration_score
