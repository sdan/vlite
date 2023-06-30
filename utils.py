import numpy as np
import torch
from torch import Tensor
import pickle

def chop_and_chunk(text, max_seq_length=128):
    '''
    Chop text into chunks of max_seq_length.
    '''

    # text can be a string or a list of strings
    if isinstance(text, str):
        # if text is a string, create a list with the string as the only element
        text = [text]
    return text


def cos_sim(a,b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    # if not isinstance(a, torch.Tensor):
    #     a = torch.tensor(a)

    # if not isinstance(b, torch.Tensor):
    #     b = torch.tensor(b)

    # if len(a.shape) == 1:
    #     a = a.unsqueeze(0)

    # if len(b.shape) == 1:
    #     b = b.unsqueeze(0)

    # a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    # b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    # return torch.mm(a_norm, b_norm.transpose(0, 1))

    product = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return product / norm