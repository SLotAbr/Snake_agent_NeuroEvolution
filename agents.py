import torch
from torch import nn


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
			nn.MaxPool(kernel_size=3, stride=3)
		)
		self.ff = nn.Linear(9, 3) #+27=96

	def forward(self, x):
		x = self.convolution(x)		
		x = x.view(x.size(0) * x.size(1))
		x = self.ff(x)	
		return x
	

class Agent(object):
	def __init__(self):
		self.body = Bare_minimum()
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.body = self.body.to(self.device)

	def __call__(self, x):
		x = x.to(self.device)
		with torch.no_grad():
			action_space = self.body.forward(x)
			return action_space
