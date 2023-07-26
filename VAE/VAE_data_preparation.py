from pickle import dump, load
from sys import path
from torch import zeros


def get_mask_replay(mask_id='1'):
	path = f'../history_buffer/VAE_data/mask_{mask_id}_replay.pkl'
	with open(path, 'rb') as f:
		history_track = load(f)
	return history_track


def prepare_arguments(game_state, head_position, direction):
	return game_state, TILE_TYPES, FIELD_SIZE, head_position, direction


if __name__ == '__main__':
	path.append('..')
	from simulation_tool import encode_game_state

	FIELD_SIZE = 9
	TILE_TYPES = {"EMPTY"  :0,
				  "BODY"   :1,
				  "FOOD"   :2,
				  "BARRIER":3}

	history_files = list(get_mask_replay(mask_id=n) for n in ['1','2','3'])
	game_states = history_files[0]+history_files[1]+history_files[2]

	rr_len = len(range(-(FIELD_SIZE-3), FIELD_SIZE-2))
	dataset = zeros(len(game_states)*2, len(TILE_TYPES), rr_len, rr_len)

	for i in range(0, len(game_states)*2, 2):
		game_state = game_states.pop()
		encoding = encode_game_state(*prepare_arguments(*game_state))
		# data augmentation
		dataset[i], dataset[i+1] = encoding, encoding.flip(dims=[3])

	with open('VAE_train_dataset.pkl','wb') as f:
		dump(dataset.unique(dim=0), f)
