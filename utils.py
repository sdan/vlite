import numpy as np
import pickle

def normalize_vectors(vectors):
    '''
    Normalizes the vectors for cosine similarity.
    '''
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]

def calculate_similarity_scores(normalized_vectors, query_vec):
    '''
    Calculates the similarity scores between the normalized vectors and the query vector.
    '''
    return np.dot(normalized_vectors, query_vec)

def get_top_k_indices(similarity_scores, top_k):
    '''
    Returns the indices of the top k most relevant vectors.
    '''
    return np.argsort(similarity_scores)[-top_k:][::-1]

def get_top_k_ids(id_to_index, top_k_indices):
    '''
    Returns the IDs of the top k most relevant vectors.
    '''
    return [k for k, v in id_to_index.items() if v in top_k_indices]

def save_data(collection, id_to_index, metadata, vectors):
    '''
    Saves the data to a file.
    '''
    with open(collection, 'wb') as f:
        pickle.dump((id_to_index, metadata, vectors), f)

def load_data(collection):
    '''
    Loads the data from a file.
    '''
    with open(collection, 'rb') as f:
        id_to_index, metadata, vectors = pickle.load(f)
    return id_to_index, metadata, vectors