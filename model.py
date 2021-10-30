import torch
import torch.nn as nn
from functools import reduce

class shave_block(nn.Module):
	def __init__(self, s):
		super(shave_block, self).__init__()
		self.s = s
	def forward(self, input):
		return input[:, :, self.s:-self.s, self.s:-self.s]
class LambdaBase(nn.Sequential):
	def __init__(self, fn, *args):
		super(LambdaBase, self).__init__(*args)
		self.lambda_func = fn

	def forward_prepare(self, input):
		output = []
		for module in self._modules.values():
			output.append(module(input))
		return output if output else input

class Lambda(LambdaBase):
	def forward(self, input):
		return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
	def forward(self, input):
		return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
	def forward(self, input):
		return reduce(self.lambda_func,self.forward_prepare(input))

class ConvBlock(nn.Module):
	expansion = 1
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
		super(ConvBlock, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
		self.bn = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
	
	def forward(self, input):
		x = self.conv(input)
		x = self.bn(x)
		x = self.relu(x)
		return x

class ConvTransBlock(nn.Module):
	expansion = 1
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
		super(ConvTransBlock, self).__init__()
		self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
		self.bn = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
	
	def forward(self, input):
		x = self.conv(input)
		x = self.bn(x)
		x = self.relu(x)
		return x

class BasicBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):#, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
		self.bn2 = nn.BatchNorm2d(out_channels)
		# self.downsample = downsample
		# self.stride = stride

	def forward(self, input):
		# identity = input

		x = self.conv1(input)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.bn2(x)

		# if self.downsample is not None:
			# identity = self.downsample(x)

		# x += identity
		# x = self.relu(x)

		return x

class NetG(nn.Module):
	def __init__(self):
		super(NetG, self).__init__()
		self.pd1 = nn.ReflectionPad2d((40, 40, 40, 40))
		self.layer1 = ConvBlock(1, 32, 9, 1, 4)
		self.layer2 = ConvBlock(32, 64, 3, 2, 1)
		self.layer3 = ConvBlock(64, 128, 3, 2, 1)
		self.layer4 = self._make_layer(2, 128, 128, 3)
		self.layer5 = self._make_layer(2, 128, 128, 3)
		self.layer6 = self._make_layer(2, 128, 128, 3)
		self.layer7 = self._make_layer(2, 128, 128, 3)
		self.layer8 = self._make_layer(2, 128, 128, 3)
		self.layer9 = ConvTransBlock(128, 64, 3, 2, 1, 1)
		self.layer10 = ConvTransBlock(64, 32, 3, 2, 1, 1)
		self.conv = nn.Conv2d(32, 2, 9, 1, 4)
		self.T = nn.Tanh()

	def _make_layer(self, s, in_channels, out_channels, kernel_size, stride=1, padding=0):
		return nn.Sequential(
			LambdaMap(lambda x: x,
				BasicBlock(in_channels, out_channels, kernel_size, stride, padding),
				shave_block(s),
			),
			LambdaReduce(lambda x,y: x+y)
		)
	
	def forward(self, input):
		x = self.pd1(input)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = self.layer6(x)
		x = self.layer7(x)
		x = self.layer8(x)
		x = self.layer9(x)
		x = self.layer10(x)
		x = self.conv(x)
		x = self.T(x)
		return x

# print(NetG())