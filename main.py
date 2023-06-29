from sentence_transformers import SentenceTransformer

import numpy as np
import pickle
import uuid
from model import EmbeddingModel

class VLite:
    '''
    vlite is a simple vector database that stores vectors in a numpy array.
    '''
    def __init__(self, collection='vlite.pkl'):
        self.collection = collection
        self.model = EmbeddingModel()
        try:
            with open(self.collection, 'rb') as f:
                self.text, self.metadata, self.vectors = pickle.load(f)
        except FileNotFoundError:
            self.text = []
            self.metadata = {}
            self.vectors = np.empty((0, 384))

    def memorize(self, data, id=None, metadata=None):
        '''
        Adds data into the database.

        Args:
            data: Text or list of texts to add to the database.
            id: Optional ID associated with the data.
            metadata: Optional dictionary of metadata associated with the data.
        '''

    def remember(self, data=None, id=None, metadata=None, top_k=2):
        '''
        Returns the top 5 most relevant vectors to the data, or matches the id or metadata.

        Args:
            data: Optional text to find similar vectors to.
            id: Optional ID associated with the data.
            metadata: Optional dictionary of metadata associated with the data.
        '''


    def save(self):
        '''
        Saves the database to a file.
        '''
        with open(self.collection, 'wb') as f:
            pickle.dump((self.id_to_index, self.metadata, self.vectors), f)


    # def remove(self, text):
    #     '''
    #     Removes the sentence from the database.
    #     '''
    #     if text in self.id_to_index:
    #         index = self.id_to_index[text]
    #         self.vectors = np.delete(self.vectors, index, 0)
    #         del self.id_to_index[text]
    #         for key, value in self.id_to_index.items():
    #             if value > index:
    #                 self.id_to_index[key] = value - 1
    #     else:
    #         print(f"Text: {text} not found in database.")
    
    # def relevancy(self, query, top_k=5):
    #     '''
    #     Returns the top_k most relevant sentences in the corpus to the query.
    #     '''
    #     query_vector = self.model.embed(query)
    #     scores = np.dot(self.vectors, query_vector)
    #     top_k_indices = np.argsort(scores)[-top_k:]
    #     top_k_texts = {k: v for k, v in self.id_to_index.items() if v in top_k_indices}
    #     return top_k_texts