"""
Potentially useful helper functions
"""
import torch


def load_data(path: str):
    """
    Function for loading MEG data from wanted file and shaping it into 2D tensor

    Returns: data: torch.Tensor
    """
    pass


def next_batch(data: torch.Tensor, batch_size: int, last: int):
    """
    Function for iterating over a tensor in mini-batches

    Returns: (mini_batch: torch.Tensor, new_last: int)
    """
    n_points = data.shape[0]
    n_dims = data.shape[1]
    ret = torch.zeros(n_points, n_dims)

    for i0 in range(batch_size):
        last = last + i0
        if last == n_points:
            last = 0

        ret[i0, :] = data[last, :]

    return ret, last


def save_embeddings(data: torch.Tensor, labels):
    pass

