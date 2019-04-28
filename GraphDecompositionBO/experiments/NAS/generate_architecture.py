import torch
import torch.nn as nn


N_NODES = 7
IN_H = IN_W = 32
IN_CH = 16


class NodeConv(nn.Module):
	def __init__(self, n_channels):
		super(NodeConv, self).__init__()
		self.bn = nn.BatchNorm2d(num_features=n_channels)
		self.relu = nn.ReLU()

	def init_parameters(self):
		pass

	def forward(self, x):
		return self.relu(self.bn(self.conv(x)))


class NodeConv3by3(NodeConv):
	def __init__(self, n_channels):
		super(NodeConv3by3, self).__init__()
		self.conv = nn.conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=True)


class NodeConv1by1(NodeConv):
	def __init__(self, n_channels):
		super(NodeConv1by1, self).__init__()
		self.conv = nn.conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, padding=0, bias=True)


class NodeMaxpool3by3(nn.Module):
	def __init__(self):
		super(NodeMaxpool3by3, self).__init__()
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

	def forward(self, x):
		return self.maxpool(x)


# Id can be considered as 1 by 1 max pool
class NodeId(nn.Module):
	def __init__(self):
		super(NodeId, self).__init__()

	def forward(self, x):
		return x


def valid_connectivity(connectivity):
	"""
	:param connectivity: np.array
	:return:
	"""
	assert connectivity.size(0) == connectivity.size(1) == N_NODES
	assert torch.sum(torch.tril(connectivity) ** 2) == 0
	queue = [0]
	while len(queue) > 0:
		curr_v = queue.pop(0)
		for i in range(curr_v + 1, N_NODES):
			if connectivity[curr_v, i] == 1:
				if i == N_NODES - 1:
					return True
				else:
					queue.append(i)
	return False


class Cell(nn.Module):
	def __init__(self, node_type, connectivity, output_node_type, n_channels):
		"""
		:param node_type: list of binary 0-1
		:param connectivity: np.array
		:param n_channels:
		"""
		assert len(node_type) == N_NODES - 2
		assert output_node_type in [0, 1]
		self.connectivity = connectivity
		self.output_node_type = output_node_type
		if output_node_type == 0: # concatenation and 1by1 conv to adjust filter size
			# TODO below is wrong, some dead node can be connected
			n_concat = int(torch.sum(connectivity[:, -1]).item())
			self.output_node = nn.Conv2d(in_channels=n_concat * n_channels, out_channels=n_channels, kernel_size=1, bias=False)
		self.node_list = [NodeConv(n_channels) if elm == 1 else NodeId() for elm in node_type]

	def forward(self, x):
		input_list = [x]
		for dst in range(1, N_NODES):
			node_input = 0
			for src in range(dst):
				node_input += input_list[src] if self.connectivity[src, dst] == 1 else None
			node_output = self.node_list[dst](node_input) if torch.sum(self.connectivity[:, dst]) > 0 else None
			input_list.append(node_output)
		if self.output_node_type == 0: # concatenation and 1by1 conv to adjust filter size (inception)
			x = self.output_node(torch.cat([input_list[i] for i in range(dst - 1) if self.connectivity[i, -1] == 1 and input_list[i] is not None], dim=1))
		else: # summation (resnet)
			x = torch.stack([input_list[i] for i in range(dst - 1) if self.connectivity[i, -1] == 1 and input_list[i] is not None], dim=0).sum(0)
		return x


class CNN(nn.Module):
	def __init__(self, node_type, connectivity):
		"""
		:param node_type: list of binary 0-1
		:param connectivity: np.array
		"""
		assert valid_connectivity(connectivity)
		self.cell1 = Cell(node_type=node_type, connectivity=connectivity, n_channels=IN_CH)
		self.maxpool1 = nn.MaxPool2d(kernel_size=2)
		self.conv1 = nn.Conv2d(in_channels=IN_CH, out_channels=IN_CH * 2, kernel_size=1, bias=False)
		self.cell2 = Cell(node_type=node_type, connectivity=connectivity, n_channels=IN_CH * 2)
		self.maxpool2 = nn.MaxPool2d(kernel_size=2)
		self.conv2 = nn.Conv2d(in_channels=IN_CH * 2, out_channels=IN_CH * 4, kernel_size=1, bias=False)
		self.cell3 = Cell(node_type=node_type, connectivity=connectivity, n_channels=IN_CH * 4)
		self.maxpool3 = nn.MaxPool2d(kernel_size=2)
		self.fc = nn.Linear(in_features=IN_CH * 4 * (IN_H / 8 * IN_W / 8), out_features=10)

	def forward(self, x):
		x = self.conv1(self.maxpool1(self.cell1(x)))
		x = self.conv2(self.maxpool2(self.cell2(x)))
		x = self.maxpool3(self.cell3(x))
		x = self.fc(x.view(x.size(0), -1))
		return x