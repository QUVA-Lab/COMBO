import torch

from COMBO.graphGP.sampler.tool_partition import group_input, ungroup_input


def neighbors(x, partition_samples, edge_mat_samples, n_vertices, uniquely=False):
    """

    :param x: 1D Tensor
    :param partition_samples:
    :param edge_mat_samples:
    :param n_vertices:
    :param uniquely:
    :return:
    """
    nbds = x.new_empty((0, x.numel()))
    for i in range(len(partition_samples)):
        grouped_x = group_input(x.unsqueeze(0), partition_samples[i], n_vertices).squeeze(0)
        grouped_nbd = _cartesian_neighbors(grouped_x, edge_mat_samples[i])
        nbd = ungroup_input(grouped_nbd, partition_samples[i], n_vertices)
        added_ind = []
        if uniquely:
            for j in range(nbd.size(0)):
                if not torch.any(torch.all(nbds == nbd[j], dim=1)):
                    added_ind.append(j)
            if len(added_ind) > 0:
                nbds = torch.cat([nbds, nbd[added_ind]])
        else:
            nbds = torch.cat([nbds, nbd])
    return nbds


def _cartesian_neighbors(grouped_x, edge_mat_list):
    """
    For given vertices, it returns all neighboring vertices on cartesian product of the graphs given by edge_mat_list
    :param grouped_x: 1D Tensor
    :param edge_mat_list:
    :return: 2d tensor in which each row is 1-hamming distance far from x
    """
    neighbor_list = []
    for i in range(len(edge_mat_list)):
        nbd_i_elm = edge_mat_list[i][grouped_x[i]].nonzero(as_tuple=False).squeeze(1)
        nbd_i = grouped_x.repeat((nbd_i_elm.numel(), 1))
        nbd_i[:, i] = nbd_i_elm
        neighbor_list.append(nbd_i)

    return torch.cat(neighbor_list, dim=0)
