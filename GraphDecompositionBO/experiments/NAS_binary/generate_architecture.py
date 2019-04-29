import toposort
import numpy as np

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
        # TODO : set init method
        pass

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class NodeConv3by3(NodeConv):
    def __init__(self, n_channels):
        super(NodeConv3by3, self).__init__(n_channels)
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=True)


class NodeConv1by1(NodeConv):
    def __init__(self, n_channels):
        super(NodeConv1by1, self).__init__(n_channels)
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, padding=0, bias=True)


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


def set_node(node_type, n_channels):
    """

    :param node_type: 1d np.array of binary value
    :param n_channels:
    :return:
    """
    assert node_type.size == 2
    if node_type[0] == 0:
        if node_type[1] == 0:
            return NodeId()
        elif node_type[1] == 1:
            return NodeMaxpool3by3()
        else:
            raise NotImplementedError
    elif node_type[0] == 1:
        if node_type[1] == 0:
            return NodeConv1by1(n_channels)
        elif node_type[1] == 1:
            return NodeConv3by3(n_channels)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def valid_net_topo(adj_mat):
    """
    :param adj_mat: 2d np.array upper triangular with zero diagonal
        (i,j) entry represents edge from i-node to j-node
    :return: topological sort, nodes are not reachable from the input node
    """
    assert adj_mat.shape[0] == adj_mat.shape[1]
    n_nodes = adj_mat.shape[0]
    assert np.sum(np.tril(adj_mat) ** 2) == 0
    aug_mat_input = np.eye(n_nodes)
    aug_mat_output = np.eye(n_nodes)
    dist_input = {v: None for v in range(1, n_nodes)}
    dist_output = {v: None for v in range(n_nodes - 1)}
    for s in range(1, n_nodes):
        aug_mat_input = np.dot(aug_mat_input, adj_mat)
        for v in np.where(aug_mat_input[0] > 0)[0]:
            dist_input[v] = s
    for s in range(1, n_nodes):
        aug_mat_output = np.dot(aug_mat_output, adj_mat.T)
        for v in np.where(aug_mat_output[-1] > 0)[0]:
            dist_output[v] = s
    if dist_input[n_nodes - 1] is None or dist_output[0] is None:
        return None
    for v in range(1, n_nodes):
        if dist_input[v] is None:
            adj_mat[v, :] = 0
            adj_mat[:, v] = 0
    for v in range(n_nodes - 1):
        if dist_output[v] is None:
            adj_mat[v, :] = 0
            adj_mat[:, v] = 0
    return adj_mat


class Cell(nn.Module):
    def __init__(self, node_type, adj_mat, n_channels):
        """
        With a valid network topology, a cell is constructed
        :param node_type: 1d np.array ((conv, pool), (1by1, 3by3)
        :param adj_mat: topological ordering, list of sets
        :param n_channels: numeric(int)
        """
        super(Cell, self).__init__()
        self.n_nodes = adj_mat.shape[0]
        assert len(node_type) == (self.n_nodes - 2) * 2
        self.adj_mat = adj_mat
        topo_input = {v: set(np.where(adj_mat[:, v] == 1)[0]) for v in range(1, self.n_nodes) if np.sum(adj_mat[:, v]) > 0}
        self.topo_order = list(toposort.toposort(topo_input))
        assert self.topo_order[0] == {0}
        assert self.topo_order[-1] == {self.n_nodes - 1}
        self.node_list = [None] + [set_node(node_type[2*i:2*i+2], n_channels) for i in range(self.n_nodes - 2)] + [NodeId()]

    def forward(self, x):
        node_output_list = [x] + [None for _ in range(1, self.n_nodes)]
        for i in range(1, len(self.topo_order)):
            for j in self.topo_order[i]:
                node_input = 0
                for k in np.where(self.adj_mat[:, j] == 1)[0]:
                    node_input += node_output_list[k]
                node_output_list[j] = self.node_list[j](node_input)
        return node_output_list[-1]


class CNN(nn.Module):
    def __init__(self, node_type, adj_mat):
        """
        :param node_type: list of binary 0-1
        :param connectadj_mativity: np.array
        """
        super(CNN, self).__init__()
        self.cell1 = Cell(node_type=node_type, adj_mat=adj_mat, n_channels=IN_CH)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels=IN_CH, out_channels=IN_CH * 2, kernel_size=1, bias=False)
        self.cell2 = Cell(node_type=node_type, adj_mat=adj_mat, n_channels=IN_CH * 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=IN_CH * 2, out_channels=IN_CH * 4, kernel_size=1, bias=False)
        self.cell3 = Cell(node_type=node_type, adj_mat=adj_mat, n_channels=IN_CH * 4)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(in_features=IN_CH * 4 * (IN_H / 8 * IN_W / 8), out_features=10)

    def forward(self, x):
        x = self.conv1(self.maxpool1(self.cell1(x)))
        x = self.conv2(self.maxpool2(self.cell2(x)))
        x = self.maxpool3(self.cell3(x))
        x = self.fc(x.view(x.size(0), -1))
        return x


if __name__ == '__main__':
    ## valid_net_topo check
    # n_nodes_ = 5
    # adj_mat_ = np.random.randint(0, 2, (n_nodes_, n_nodes_))
    # adj_mat_ -= np.tril(adj_mat_)
    # print(adj_mat_)
    # adj_mat_ = valid_net_topo(adj_mat_)
    # print(adj_mat_)
    # topo_input_ = {v_: set(np.where(adj_mat_[:, v_] == 1)[0]) for v_ in range(1, n_nodes_) if np.sum(adj_mat_[:, v_]) > 0}
    # topo_order_ = toposort.toposort(topo_input_)
    # print(list(topo_order_))
    ## Cell test
    n_nodes_ = 5
    n_ch_ = 32
    adj_mat_ = None
    while adj_mat_ is None:
        adj_mat_ = np.random.randint(0, 2, (n_nodes_, n_nodes_))
        adj_mat_ -= np.tril(adj_mat_)
        adj_mat_ = valid_net_topo(adj_mat_)
    node_type_ = np.random.randint(0, 2, (2 * (n_nodes_ - 2)))
    cell_ = Cell(node_type_, adj_mat_, n_channels=n_ch_)
    input_data_ = torch.randn(3, n_ch_, 32, 32)
    output_data_ = cell_(input_data_)
