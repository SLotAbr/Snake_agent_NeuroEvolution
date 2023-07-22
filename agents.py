import torch
from torch import nn, Tensor
from copy import deepcopy
from collections import deque
from random import randint


class Bare_minimum(torch.nn.Module):
	def __init__(self):
		super(Bare_minimum, self).__init__()
		self.convolution = nn.Sequential(
			nn.Conv2d(in_channels=5, out_channels=2, kernel_size=1, padding=0), #12
			nn.Sigmoid(),
			nn.ReLU(),
			nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=0), #+19*2=50
			nn.Sigmoid(),
			nn.ReLU(),
			nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=0), #+19=69
			nn.Sigmoid(),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=3)
		)
		self.ff = nn.Linear(9, 3) #+30=99

	def forward(self, x):
		x = self.convolution(x)		
		x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
		x = self.ff(x)	
		return x
	

class Agent(object):
	def __init__(self, agent_name):
		AGENTS = {'Bare_minimum':Bare_minimum()}
		self.body = AGENTS[agent_name]
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.body = self.body.to(self.device)
		self.state_dict_info = deepcopy(self.body.state_dict())
		self.genome_length=0
		# key: (parameters_number, exact_tensor_size)
		for key in self.state_dict_info:
			parameters_number = self.state_dict_info[key].view(-1).shape[0]
			self.genome_length+=parameters_number
			self.state_dict_info[key] = \
				(parameters_number, self.state_dict_info[key].shape)

	def create_generator(self):
		ids = range(self.genome_length)
		for i in ids:
			yield i

	def set_genome(self, genome):
		generator = self.create_generator()
		new_state_dict = deepcopy(self.state_dict_info)
		for key in new_state_dict:
			item = Tensor(
				[genome[next(generator)] for _ in range(new_state_dict[key][0])]
			)
			item = item.view(new_state_dict[key][1])
			new_state_dict[key] = item
		self.body.load_state_dict(new_state_dict)

	def __call__(self, x):
		x = x.to(self.device)
		with torch.no_grad():
			action_space = self.body.forward(x)
			return torch.argmax(action_space)-1


class Rule_based_agent(object):
	"""Rule_based_agent could be used for 9x9 field only"""
	def __init__(self, mask_id):
		if mask_id=='1':
			self.mask = deque([
				((1,1), -1),
				((1,7), -1),
				((2,7), -1),
				((2,2), 1),
				((3,2), 1),
				((3,7), -1),
				((4,7), -1),
				((4,2), 1),
				((5,2), 1),
				((5,7), -1),
				((7,7), -1),
				((7,6), -1),
				((6,6), 1),
				((6,5), 1),
				((7,5), -1),
				((7,4), -1),
				((6,4), 1),
				((6,3), 1),
				((7,3), -1),
				((7,2), 'choice')
			])
		elif mask_id=='2':
			pass
			# self.mask = 
		elif mask_id=='3':
			pass
			# self.mask = 
		else:
			raise ValueError

		self.mask_id = mask_id
		self.chosen_route = []
		self.position, self.action = self.get_condition((-1,-1))

	def choice(self, food_position):
		if self.mask_id=='1':
			ROUTES = {-1: deque([((6,2), 1), 
								 ((6,1), -1)]), 
					  0 : deque([((7, 1), -1)])}
			if food_position in [(6,2), (7,1)]:
				if food_position in [(6,2)]:
					action = -1
				else:
					action = 0
			else:
				action = randint(-1,0)
			self.chosen_route = ROUTES[action]
			return action

		elif self.mask_id=='2':
			pass
		elif self.mask_id=='3':
			pass
		else:
			raise ValueError
	
	def get_condition(self, food_position):
		if not self.chosen_route:
			condition = self.mask.popleft()
			self.mask.append(condition)
			pos, act = condition
			if act=='choice':
				act = self.choice(food_position)
		else:
			pos, act = self.chosen_route.popleft()
		return pos, act

	def get_action(self, head_position, food_position):
		if head_position not in [self.position]:
			return 0
		else:
			output = self.action
			self.position, self.action = self.get_condition(food_position)
			return output
