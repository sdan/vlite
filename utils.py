import numpy as np
import torch
from torch import Tensor
import pickle

def chop_and_chunk(text, max_seq_length=128):
    # text can be a string or a list of strings
    if isinstance(text, str):
        # if text is a string, create a list with the string as the only element
        text = [text]
    return text


def cos_sim(vec,mat):
    sim = vec @ mat.T
    sim /= np.linalg.norm(vec) * np.linalg.norm(mat, axis=1)
    return sim