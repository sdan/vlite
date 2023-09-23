from .utils import chop_and_chunk, cos_sim
from .model import EmbeddingModel
from typing import Any
from uuid import uuid4
import numpy as np
import datetime
import warnings

class VLite:
    '''
    vlite is a simple vector database that stores vectors in a numpy array.
    '''
    def __init__(self, collection:str=None, device:str='mps', model_name:str=None):
        """
        Initialize a new VLite database.

        Parameters:
        collection (str): The filename to save the database to.
        device (str): The device to run the model on. Defaults to 'mps'.
        model_name (str): The name of the model to use. Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
        """

		# Filename must be unique between runs. Saving to the same file will append vectors to previous run's vectors
        if collection is None:
            current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            collection = f"vlite_{current_datetime}.npz"
            
        self._collection = collection
        self._device = device
        self._model = EmbeddingModel(model_name)
        try:
            with np.load(self._collection, allow_pickle=True) as data:
                self._data = data['texts'].tolist()
                self._metadata = data['metadata'].tolist()
                self._vectors = data['vectors']
        except FileNotFoundError:
            self._data = []
            self._metadata = {}
            self._vectors = np.empty((0, self._model.dimension))
    
    def add_vector(self, vector:Any):
        """
        Add a vector to the database.

        Parameters:
        vector (Any): The vector to add to the database.
        """
        self._vectors = np.vstack((self._vectors, vector))

    def get_similar_vectors(self, vector:Any, top_k:int=5, DEBUG:bool=False):
        """
        Retrieve the most similar vectors to a given vector.

        Parameters:
        vector (Any): The vector to search for.
        top_k (int): The number of results to return with the highest similarity.
        DEBUG (bool): Print debug information. Repo maintainer use only.
        """
        sims = cos_sim(vector, self._vectors)
        sims = sims[0]
        if DEBUG:
            print("[get_similar_vectors] Sims:", sims.shape)

        top_k_idx = np.argsort(sims)[::-1][:top_k]
        if DEBUG:
            print("[get_similar_vectors] Top k idx:", top_k_idx)
            print("[get_similar_vectors] Top k sims:", sims[top_k_idx])

        return top_k_idx, sims[top_k_idx]

    def memorize(self, text: str, id: Any=None, metadata: Any=None):
        """
        Add a text to the database.

        Parameters:
        text (str): The text to add to the database.
        id (str): The id of the text to add to the database.
        metadata (Any): Any metadata to associate with the text.
        """
        id = id or str(uuid4())
        chunks = chop_and_chunk(text)
        encoded_data = self._model.embed(texts=chunks, device=self._device)
        self._vectors = np.vstack((self._vectors, encoded_data))
        ingest_text_chunks(chunks, self, metadata, id)
        self.save()
        return id, self._vectors

    def remember(self, text:str=None, id:Any=None, top_k:int=5, DEBUG:bool=False):
        """
        Retrieve a text from the database by id or by text.

        Parameters:
        text (str): The text to search for.
        id (str): The id of the text to search for.
        top_k (int): The number of results to return with the highest similarity.
        DEBUG (bool): Print debug information. Repo maintainer use only.
        """
        if id:
            return self._metadata[id]
        if text:

            sims = cos_sim(self._model.embed(texts=text, device=self._device) , self._vectors)
            if DEBUG:
                print("[remember] Sims:", sims.shape)
                
            sims = sims[0]

			# top_k cannot be higher than the number of similarities returned
            top_k = min(top_k, len(sims))

            # Use np.argpartition to partially sort only the top k values
            top_k_idx = np.argpartition(sims, -top_k)[-top_k:]  

            # Use np.argsort to sort just those top k indices
            top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]  

            if DEBUG:
                print("[remember] Top k sims:", sims[top_k_idx])

            return [self._data[idx] for idx in top_k_idx], sims[top_k_idx]
            
    def save(self):
        """Save the database to disk."""
        with open(self._collection, 'wb') as f:
            np.savez(f, texts=self._data, metadata=self._metadata, vectors=self._vectors)

    @property
    def collection(self):
        """The filename of the database."""
        return self._collection
    
    @property
    def device(self):
        """The device the model is running on."""
        return self._device
    
    @property
    def model(self):
        """The model used to generate vectors."""
        return self._model

    @property
    def texts(self):
        """The texts in the database. Deprecated. Use VLite.data instead."""
        warnings.warn("VLite.texts is deprecated. Use VLite.data instead.", DeprecationWarning)
        return self._data

    @property
    def data(self):
        """Data stored in the database."""
        return self._data
    
    @property
    def metadata(self):
        """Metadata stored in the database."""
        return self._metadata

    @property
    def vectors(self):
        """Vectors stored in the database."""
        return self._vectors
    

"""
---------------------
--- Utility Code ----
---------------------
"""

def ingest_text_chunks(chunks, db: VLite, metadata = None, key = None):
    """Ingest text chunks into the database"""
    text_key = key or len(db.texts) - 1
    for chunk in chunks:
        db.data.append(chunk)
        db.metadata[text_key] = metadata or {}
        db.metadata[text_key]['index'] = text_key