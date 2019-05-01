import torch


def init_architectures():
	"""
	Assuming 7 nodes
	:return:
	"""
	architecture_list = []

	# 1
	# line graph
	# Conv3, Id, Id, Id, Id
	node_type = torch.LongTensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
	outbounds = [torch.LongTensor([1, 0, 0, 0, 0, 0]),
	             torch.LongTensor([   1, 0, 0, 0, 0]),
	             torch.LongTensor([      1, 0, 0, 0]),
	             torch.LongTensor([         1, 0, 0]),
	             torch.LongTensor([            1, 0]),
	             torch.LongTensor([               1])]
	architecture_list.append(torch.cat([node_type] + outbounds))

	# 2
	# line graph
	# Conv3, Conv3, Conv3, Conv3, Conv3
	node_type = torch.LongTensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
	outbounds = [torch.LongTensor([1, 0, 0, 0, 0, 0]),
	             torch.LongTensor([   1, 0, 0, 0, 0]),
	             torch.LongTensor([      1, 0, 0, 0]),
	             torch.LongTensor([         1, 0, 0]),
	             torch.LongTensor([            1, 0]),
	             torch.LongTensor([               1])]
	architecture_list.append(torch.cat([node_type] + outbounds))

	# 3
	# id + line graph
	# Id, Conv3, Conv3, Conv3, Conv3
	node_type = torch.LongTensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
	outbounds = [torch.LongTensor([1, 0, 0, 0, 0, 1]),
	             torch.LongTensor([   1, 0, 0, 0, 0]),
	             torch.LongTensor([      1, 0, 0, 0]),
	             torch.LongTensor([         1, 0, 0]),
	             torch.LongTensor([            1, 0]),
	             torch.LongTensor([               1])]
	architecture_list.append(torch.cat([node_type] + outbounds))

	# 4
	# id + line graph
	# Id, Conv3, MaxPool3, Conv3, MaxPool3
	node_type = torch.LongTensor([0, 0, 1, 1, 0, 1, 1, 1, 0, 1])
	outbounds = [torch.LongTensor([1, 0, 0, 0, 0, 1]),
	             torch.LongTensor([   1, 0, 0, 0, 0]),
	             torch.LongTensor([      1, 0, 0, 0]),
	             torch.LongTensor([         1, 0, 0]),
	             torch.LongTensor([            1, 0]),
	             torch.LongTensor([               1])]
	architecture_list.append(torch.cat([node_type] + outbounds))

	# 5
	# id + 2 paths
	# Id, Conv3, Conv3, Conv3, Conv3
	node_type = torch.LongTensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
	outbounds = [torch.LongTensor([1, 1, 1, 0, 0, 0]),
	             torch.LongTensor([   0, 0, 0, 0, 1]),
	             torch.LongTensor([      1, 0, 0, 0]),
	             torch.LongTensor([         0, 0, 1]),
	             torch.LongTensor([            1, 0]),
	             torch.LongTensor([               1])]
	architecture_list.append(torch.cat([node_type] + outbounds))

	# 6
	# id + 2 paths
	# Id, Conv3, MaxPool3, Conv3, MaxPool3
	node_type = torch.LongTensor([0, 0, 1, 1, 0, 1, 1, 1, 0, 1])
	outbounds = [torch.LongTensor([1, 1, 1, 0, 0, 0]),
	             torch.LongTensor([   0, 0, 0, 0, 1]),
	             torch.LongTensor([      1, 0, 0, 0]),
	             torch.LongTensor([         0, 0, 1]),
	             torch.LongTensor([            1, 0]),
	             torch.LongTensor([               1])]
	architecture_list.append(torch.cat([node_type] + outbounds))

	# 7
	# conv + 2 path
	# Conv3, Conv3, Conv3, Conv3, Conv3
	node_type = torch.LongTensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
	outbounds = [torch.LongTensor([1, 1, 1, 0, 0, 0]),
	             torch.LongTensor([   0, 0, 0, 0, 1]),
	             torch.LongTensor([      1, 0, 0, 0]),
	             torch.LongTensor([         0, 0, 1]),
	             torch.LongTensor([            1, 0]),
	             torch.LongTensor([               1])]
	architecture_list.append(torch.cat([node_type] + outbounds))

	# 8
	# conv + 2 paths
	# Conv3, Conv3, MaxPool3, Conv3, MaxPool3
	node_type = torch.LongTensor([1, 1, 1, 1, 0, 1, 1, 1, 0, 1])
	outbounds = [torch.LongTensor([1, 1, 1, 0, 0, 0]),
	             torch.LongTensor([   0, 0, 0, 0, 1]),
	             torch.LongTensor([      1, 0, 0, 0]),
	             torch.LongTensor([         0, 0, 1]),
	             torch.LongTensor([            1, 0]),
	             torch.LongTensor([               1])]
	architecture_list.append(torch.cat([node_type] + outbounds))

	# 9
	# complete graph
	# Conv3, Conv3, Conv3, Conv3, Conv3
	node_type = torch.LongTensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
	outbounds = [torch.LongTensor([1, 1, 1, 1, 1, 1]),
	             torch.LongTensor([   1, 1, 1, 1, 1]),
	             torch.LongTensor([      1, 1, 1, 1]),
	             torch.LongTensor([         1, 1, 1]),
	             torch.LongTensor([            1, 1]),
	             torch.LongTensor([               1])]
	architecture_list.append(torch.cat([node_type] + outbounds))

	# 10
	# complete graph
	# Conv3, MaxPool3, Conv3, MaxPool3, Conv3
	node_type = torch.LongTensor([1, 1, 0, 1, 1, 1, 0, 1, 1, 1])
	outbounds = [torch.LongTensor([1, 1, 1, 1, 1, 1]),
	             torch.LongTensor([   1, 1, 1, 1, 1]),
	             torch.LongTensor([      1, 1, 1, 1]),
	             torch.LongTensor([         1, 1, 1]),
	             torch.LongTensor([            1, 1]),
	             torch.LongTensor([               1])]
	architecture_list.append(torch.cat([node_type] + outbounds))

	return torch.stack(architecture_list, dim=0)


if __name__ == "__main__":
	print(init_architectures())