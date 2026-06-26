from torch import nn

import torch

class MLPRegressor(nn.Module):
	"""
	Simple MLP for ozone regression.
	"""
	
	def __init__(self, layers):
		super().__init__()
	
		self.activation = nn.LeakyReLU()
		self.loss_function = nn.MSELoss(reduction="mean")
		self.linears = nn.ModuleList(
			[nn.Linear(int(layers[i]), int(layers[i + 1])) for i in range(len(layers) - 1)]
		)
	
		print(self.linears)
	
		for layer in self.linears:
			nn.init.xavier_normal_(layer.weight.data, gain=1.0)
			nn.init.zeros_(layer.bias.data)
	
	def forward(self, x):
		if not torch.is_tensor(x):
			x = torch.from_numpy(x)
	
		a = x
		for i in range(len(self.linears) - 1):
			a = self.activation(self.linears[i](a))
	
		a = self.linears[-1](a)
		return a
	
	def loss(self, y_hat, y):
		"""
		Calculate the loss between model prediction and true target.
	
		Parameters
		----------
		y_hat : torch.Tensor
			Model prediction.
	
		y : torch.Tensor
			True target label.
	
		Returns
		-------
		loss : torch.Tensor
			Training loss.
		"""
		return self.loss_function(y_hat, y)
        
