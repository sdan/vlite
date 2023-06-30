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

class VLite:
    '''
    vlite is a simple vector database that stores vectors in a numpy array.
    '''
    def __init__(self, collection='vlite.pkl'):
        self.collection = collection
        self.model = EmbeddingModel()
        # try:
        #     with open(self.collection, 'rb') as f:
        #         self.data = pickle.load(f)
        # except FileNotFoundError:

        self.texts = []
        self.metadata = {}
        self.vectors = np.empty((0, 384))

    def memorize(self, text, id=None, metadata=None):
        if not id:
            id = str(uuid4())
        chunks = chop_and_chunk(text)
        for chunk in chunks:
            encoded_data = self.model.embed(chunk)

            print("[+] Encoded:", encoded_data.shape)

            self.texts.append(chunk)
            self.metadata[len(self.texts) - 1] = metadata or {}
            self.metadata[len(self.texts) - 1]['index'] = len(self.texts) - 1
            
            self.vectors = np.vstack((self.vectors, encoded_data))   
        
        print("[+] Memorizing with ID:", id)

        print("[Done]")

    def remember(self, text=None, id=None, top_k=2):
        if id:
            return self.metadata[id]
        
        if text:
            query = self.model.embed(text) 

            corpus = self.vectors

            print("[+] Query shape:", query.shape)

            sims = cos_sim(query, corpus)

            print("[+] Similarities:", sims)

            sims = sims.flatten()

            top_3_idx = np.argsort(sims)[::-1][:3]
            print("[+] Top 3 indices:", top_3_idx)

            # iterate over the top 3 most similar sentences
            for idx in top_3_idx:
                print("[+] Index:", idx)
                print("[+] Sentence:", self.texts[idx])
                print("[+] Metadata:", self.metadata[idx])

            
                  
    # def remember_bench(self, query, corpus):

    #     query_embedding = self.embedder.encode(query)
    #     corpus_embeddings = self.embedder.encode(corpus)

    #     cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    #     print("[+] Cos scores:", cos_scores)

    #     hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)

    #     print("[+] Hits:", hits)    
            
    def save(self):
        with open(self.collection, 'wb') as f:
            pickle.dump(self.data, f)
