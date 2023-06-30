import numpy as np
import pickle
import uuid
from model import EmbeddingModel
import torch

from utils import chop_and_chunk, cos_sim

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from uuid import uuid4
import pickle
import torch

from sentence_transformers import SentenceTransformer, util

class VLite:
    '''
    vlite is a simple vector database that stores vectors in a numpy array.
    '''
    def __init__(self, collection='vlite.pkl'):
        self.collection = collection
        self.model = EmbeddingModel()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        # try:
        #     with open(self.collection, 'rb') as f:
        #         self.data = pickle.load(f)
        # except FileNotFoundError:
        self.data = {}

    def memorize(self, text, id=None, metadata=None):
        if not id:
            id = str(uuid4())
        chunks = chop_and_chunk(text)
        for chunk in chunks:
            encoded_data = self.model.embed(chunk)
            encoded_data_bench = self.embedder.encode(chunk)

            print("[+] Encoded:", encoded_data.shape)
            print("[+] Bench:", encoded_data_bench.shape)
            
            self.data[id] = {'text': text, 'vectors': encoded_data, 'metadata': metadata}
        
        print("[+] Memorizing with ID:", id)

        print("[Done]")

    def remember(self, text=None, id=None, top_k=2):
        if id:
            return self.data[id]
        
        if text:
            query = self.model.embed(text) 

            corpus = [self.data[i]['vectors'] for i in self.data]
            # for i in self.data:
            #     print("[+] seldata:", self.data[i]['vectors'])

            # print shape of query and corpus
            print("[+] Query shape:", query.shape)

            # Initialize an empty list to store the similarities
            sims = []
            # Compute the cosine similarity between the query and each vector in the corpus
            for vector in corpus:
                vector = vector.flatten()  # Flatten the 2D array into a 1D array
                sims.append(cos_sim(query, vector))

            print("[+] Similarities:", sims)
            
                  
    def remember_bench(self, text=None):

        query_embedding = self.model.embed(text)

        corpus = [self.data[i]['vectors'].flatten() for i in self.data]

        hits = util.semantic_search(query_embedding, corpus, top_k=5)
        hits = hits[0]      #Get the hits for the first query
        for hit in hits:
            print("(Score: {:.4f})".format(hit['score']))
            
    def save(self):
        with open(self.collection, 'wb') as f:
            pickle.dump(self.data, f)



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