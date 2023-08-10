import torch
import torch.nn as nn


class Encoder(nn.Module):
	def __init__(self, latent_space_shape=16):
		super(Encoder, self).__init__()
		self.latent_space_shape = latent_space_shape
		# Hout =(Hin +2*padding[0]−(kernel_size[0]−1)−1)/stride[0] +1
		self.convolution = nn.Sequential(
			nn.Conv2d(in_channels=4, out_channels=15, kernel_size=3, stride=2, padding=0), # 13 -> 6
			nn.LeakyReLU(),
			nn.Conv2d(in_channels=15, out_channels=20, kernel_size=3, stride=1, padding=0), # 4
			nn.LeakyReLU(),
			nn.Conv2d(in_channels=20, out_channels=10, kernel_size=2, stride=2, padding=0), # 2
			nn.LeakyReLU(),
		)
		self.mu = nn.Linear(2*2*10, latent_space_shape)
		self.logvar = nn.Linear(2*2*10, latent_space_shape)

	def forward(self, x):
		x = self.convolution(x)
		x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
		mu, logvar = self.mu(x), self.logvar(x)
		sigma = logvar.exp()
		noise = torch.randn(self.latent_space_shape).to(
			torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		)
		# Kullback-Leibler Divergence
		self.KLD = -0.5 * (1 + logvar - mu.square() - sigma).sum()
		return mu + sigma * noise


class Decoder(nn.Module):
	def __init__(self, latent_space_shape=16):
		super(Decoder, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(latent_space_shape, 2*2*10),
			nn.LeakyReLU()
		)
		self.convtranspose = nn.Sequential(
			nn.ConvTranspose2d(in_channels=10, out_channels=20, kernel_size=2, stride=2, padding=0),
			nn.LeakyReLU(),
			nn.ConvTranspose2d(in_channels=20, out_channels=15, kernel_size=3, stride=1, padding=0),
			nn.LeakyReLU(),
			nn.ConvTranspose2d(in_channels=15, out_channels=4, kernel_size=3, stride=2, padding=0)
		)

	def forward(self, x):
		x = self.fc(x)
		x = x.view(1, 10, 2, 2) # (N, c_in, H, W)
		return self.convtranspose(x)


class VariationalAutoencoder(nn.Module):
	def __init__(self, latent_space_shape=16):
		super(VariationalAutoencoder, self).__init__()
		self.encoder = Encoder(latent_space_shape)
		self.decoder = Decoder(latent_space_shape)

	def forward(self, x):
		x = self.encoder(x)
		return self.decoder(x)


def train(vae, encoded_game_states_pool, epochs=20):
	N = len(encoded_game_states_pool)
	item_shape = encoded_game_states_pool.shape
	item_shape = (1, item_shape[1], item_shape[2], item_shape[3])
	loss_history = 'epoch, epoch_loss\n'
	
	optimizer = torch.optim.Adam(vae.parameters())
	for epoch in range(epochs):
		epoch_loss = 0
		train_sequence = torch.randperm(N)
		for i in train_sequence:
			optimizer.zero_grad()
			item = encoded_game_states_pool[i].view(item_shape).to(device)
			restored_item = vae(item)
			
			loss = ((item - restored_item)**2).sum() + vae.encoder.KLD
			epoch_loss += loss
			loss.backward()

			optimizer.step()
		loss_history += f'{epoch:3d}:{epoch_loss}\n'

	with open('loss_history.txt', 'w') as f:
		f.write(loss_history)
	return vae


def compression_evaluation(model, item):
	item = item.view(1,4,13,13)
	with torch.no_grad():
		restored_item = model(item)
		return (item-restored_item).abs().sum()/item.sum()


if __name__ == '__main__':
	with open('VAE_train_dataset.pkl', 'rb') as f:
		# torch.Size([4280, 4, 13, 13])
		encoded_game_states_pool = load(f)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	vae = VariationalAutoencoder().to(device)
	encoded_game_states_pool = encoded_game_states_pool.to(device)
	vae = train(vae, encoded_game_states_pool, epochs=20)
	torch.save(vae.encoder, 'VAE_model.pt')
