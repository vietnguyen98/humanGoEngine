import torch
import torch.nn as nn
import torch.nn.functional as F

class OutputLayer(nn.Module):
	# 1v1 conv + symmetric bias
	def __init__(self, in_channels):
		super(OutputLayer, self).__init__()
		self.biasMap = torch.zeros(19, 19)
		self.conv = nn.Conv2d(in_channels, 1, 1, bias = False)
		self.flatten = nn.Flatten()
		counter = 0
		for i in range(19):
			for j in range(19):
				if self.biasMap[i][j] != 0:
				    continue
				self.biasMap[i][j] = counter
				self.biasMap[18 - i][j] = counter
				self.biasMap[i][18 - j] = counter
				self.biasMap[18 - i][18 - j] = counter
				self.biasMap[j][i] = counter
				self.biasMap[j][18 - i] = counter
				self.biasMap[18 - j][i] = counter
				self.biasMap[18 - j][18 - i] = counter
				counter += 1
		self.bias = nn.Parameter(torch.zeros(55))

	def forward(self, x):
		# x: (batch_size, in_channels, 19, 19) => (batch_size, 1, 19, 19)
		x = self.conv(x)
		for i in range(19):
			for j in range(19):
				x[i][j] += self.bias[self.biasMap[i][j] - 1]
		x = self.flatten(x)
		return x

class symConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(symConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # only supports kernel sizes of 3x3 and 5x5 rn
        assert kernel_size == 3 or kernel_size == 5
        if kernel_size == 3:
            self.padding = 1
            self.uniqueWeights = 3
            self.weightMap = [[1, 2, 1], [2, 0, 2], [1, 2, 1]]
        elif kernel_size == 5:
            self.padding = 2
            self.uniqueWeights = 6
            self.weightMap = [[1, 3, 2, 3, 1], [3, 4, 5, 4, 3], [2, 5, 0, 5, 2], [3, 4, 5, 4, 3], [1, 3, 2, 3, 1]]
        #k = np.sqrt(1 / (in_channels * kernel_size ** 2))
        #self.weight = nn.Parameter((torch.rand((out_channels, in_channels, self.uniqueWeights)) * 2 * k) - k)
        self.weight = nn.Parameter(torch.ones((out_channels, in_channels, self.uniqueWeights)))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        # input size: b, c, h, w
            
    def forward(self, x):
        convFilter = torch.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                convFilter[:, :, i, j] = self.weight[:, :, self.weightMap[i][j]]
        return F.conv2d(x, convFilter, padding = self.padding, bias = self.bias)