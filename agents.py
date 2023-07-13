import torch
from torch import nn, Tensor
from copy import deepcopy


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
	def __init__(self):
		self.body = Bare_minimum()
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
			return torch.argmax(action_space)
