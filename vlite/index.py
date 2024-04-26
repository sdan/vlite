import numpy as np
import struct
import json
from enum import Enum
from typing import List, Union, Dict

# not implemented
class BinaryVectorIndex:
    def __init__(self, embedding_size=64):
        self.index = {}
        self.embedding_size = embedding_size
        
    def add(self, chunk_id, binary_vector):
        binary_vector = binary_vector.tolist()
        self.index[chunk_id] = binary_vector
    
    def add_batch(self, chunk_ids, binary_vectors):
        for chunk_id, binary_vector in zip(chunk_ids, binary_vectors):
            self.add(chunk_id, binary_vector)
        
    def remove(self, chunk_id):
        if chunk_id in self.index:
            del self.index[chunk_id]

    def search(self, query_vector, top_k):
        query_vector = np.array(query_vector.tolist())
        binary_vectors = np.array(list(self.index.values()))
        chunk_ids = np.array(list(self.index.keys()))

        distances = np.count_nonzero(binary_vectors != query_vector[:binary_vectors.shape[1]], axis=1)
        similarities = 1 - distances / binary_vectors.shape[1]

        sorted_indices = np.argsort(similarities)[::-1]
        top_k_indices = sorted_indices[:top_k]

        top_k_ids = chunk_ids[top_k_indices]
        top_k_scores = similarities[top_k_indices]

        return top_k_ids.tolist(), top_k_scores.tolist()