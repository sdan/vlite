from .utils import chop_and_chunk, cos_sim
from typing import Any, List, Tuple, Union
from .model import EmbeddingModel
import numpy as np
import datetime
import warnings
import uuid
import traceback

class Data:
    """Generic data class for vector storage with property special property access."""

    def __init__(self, data:dict=None):
        """Initialize a new Data object."""
        if data is None:
            data = {}
        self._data = data
    
    def __getitem__(self, key: str):
        """Get a value from the data object. Key must be a string."""
        key = str(key)
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any):
        """Set a value in the data object. Key must be a string."""
        key = str(key)
        self._data[key] = value

    def __add__(self, data: Union[dict, 'Data']):
        """Add a dict or Data object to the data object."""
        if isinstance(data, dict):
            self._data.update(data)
        elif isinstance(data, Data):
            self._data.update(data._data)
        else:
            raise TypeError("Addition must be a dict or Data object.")
        return self

    def __len__(self):
        """Return the length of the data object."""
        return len(self._data)
    
    def __delitem__(self, key: str) -> None:
        """Delete an item from the data object."""
        del self._data[key]
    
    def append(self, value: Any):
        keys = list(self._data.keys())
        str_int_list = list(map(str, range(len(keys))))
        if keys == str_int_list:
            key = str(len(self._data))
            self._data[key] = value
        else:
            raise ValueError("Keys are not sequential. Cannot append value. Set a key manually instead.")
    
    def keys(self):
        """Return the keys of the data object."""
        return self._data.keys()

    def values(self):
        """Return the values of the data object."""
        return self._data.values()

class VLite:
    '''
    vlite is a simple vector database that stores vectors in a numpy array.
    '''
    _collection = None
    _device = None
    _model = None
    _data = Data()
    _metadata = Data()
    _vectors = None
    _vector_key_store = []
    _info = None

    def __init__(self, collection:str=None, device:str='mps', model_name:str=None, info:dict=None, DEBUG:bool=False):
        """
        Initialize a new VLite database.

        Parameters:
        collection (str): The filename to save the database to.
        device (str): The device to run the model on. Defaults to 'mps'.
        model_name (str): The name of the model to use. Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
        """
        self.DEBUG = DEBUG
	    # Filename must be unique between runs. Saving to the same file will append vectors to previous run's vectors
        if collection is None:
            current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            collection = f"vlite_{current_datetime}.npz"
            
        self.collection = collection
        self.device = device
        self.model = EmbeddingModel(model_name)
        try:
            with np.load(self.collection, allow_pickle=True) as data:
                self.data = data['texts'].tolist()
                self.metadata = data['metadata'].tolist()
                self.vectors = data['vectors']
                self.info = data["info"].tolist()
                self._vector_key_store = list(self.data.keys())
        except FileNotFoundError:
            self.data = Data()
            self.metadata = Data()
            self.vectors = np.empty((0, self.model.dimension))
            self.info = info
            self._vector_key_store = []
    
    def add_vector(self, vector:Any):
        """
        Add a vector to the database.

        Parameters:
        vector (Any): The vector to add to the database.
        """
        self.vectors = np.vstack((self.vectors, vector))

    def get_similar_vectors(self, vector:Any, top_k:int=5, DEBUG:bool=False):
        """
        Retrieve the most similar vectors to a given vector.

        Parameters:
        vector (Any): The vector to search for.
        top_k (int): The number of results to return with the highest similarity.
        DEBUG (bool): Print debug information. Repo maintainer use only.
        """
        sims = cos_sim(vector, self.vectors)
        sims = sims[0]
        if DEBUG:
            print("[get_similar_vectors] Sims:", sims.shape)

        top_k_idx = np.argsort(sims)[::-1][:top_k]
        if DEBUG:
            print("[get_similar_vectors] Top k idx:", top_k_idx)
            print("[get_similar_vectors] Top k sims:", sims[top_k_idx])

        return top_k_idx, sims[top_k_idx]

    def memorize(self, text: str, id: Any=None, metadata: Any=None) -> Tuple[str, List[float]]:
        """
        Add a text to the database.

        Parameters:
        text (str): The text to add to the database.
        id (str): The id of the text to add to the database.
        metadata (Any): Any metadata to associate with the text.
        """
        if id != None:
            id = str(id)
        else:
            id = uuid.uuid4()
        
        encoded_data = self.model.embed(texts=text, device=self.device)
        self.vectors = np.vstack((self.vectors, encoded_data))
        self._vector_key_store.append(id)
        add_data(text, self, metadata, id)
        self.save()
        return id, encoded_data[0]

    def remember(self, text:str=None, id:Any=None, top_k:int=5, DEBUG:bool=False):
        """
        Retrieve a text from the database by id or by text.

        Parameters:
        text (str): The text to search for.
        id (str): The id of the text to search for.
        top_k (int): The number of results to return with the highest similarity.
        DEBUG (bool): Print debug information. Repo maintainer use only.

        Returns:
        data (List[str]): The text(s) retrieved from the database.
        metadata (List[Any]): The metadata associated with the text(s).
        similiarities (List[float]): The similarity score(s) of the text(s) to the query.
        """
        if id is not None:
            return self.data[id], self.metadata[id], None
        
        if text is not None:
            sims = cos_sim(self.model.embed(texts=text, device=self.device) , self.vectors)
            if DEBUG:
                print("[remember] Vectors:", self.vectors.shape)
                print("[remember] Sims:", sims.shape)
                
            sims = sims[0]

			# top_k cannot be higher than the number of similarities returned
            top_k = min(top_k, len(sims))

            # Use np.argpartition to partially sort only the top k values
            top_k_idx = np.argpartition(sims, -top_k)[-top_k:]  

            # Use np.argsort to sort just those top k indices
            if DEBUG:
                print(f'remember top_k_idx pre sort {top_k_idx}')
            top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]
            if DEBUG:
                print(f'remember top_k_idx post sort {top_k_idx}')
            top_k_keys = [self._vector_key_store[idx] for idx in top_k_idx]

            if DEBUG:
                print("[remember] Top k sims:", sims[top_k_idx])
            
            data = [self.data[key] for key in top_k_keys]
            metadata = [self.metadata[key] for key in top_k_keys]
            similiarities = sims[top_k_idx]
            return data, metadata, similiarities
    
    def forget(self, id: str):
        """Delete an entry from the database by id."""
        del self.data[id]
        del self.metadata[id]
        self.vectors = np.delete(self.vectors, self._vector_key_store.index(id), 0)
        self._vector_key_store.remove(id)
        self.save()
            
    def save(self):
        """Save the database to disk."""
        with open(self.collection, 'wb') as f:
            np.savez(
                        f, 
                        texts=self.data, 
                        metadata=self.metadata, 
                        vectors=self.vectors,
                        info=self.info
                    )

    @property
    def collection(self):
        """The filename of the database."""
        return self._collection
    
    @collection.setter
    def collection(self, value):
        """The filename of the database."""
        self._collection = value
    
    @property
    def device(self):
        """The device the model is running on."""
        return self._device
    
    @device.setter
    def device(self, value):
        """The device the model is running on."""
        self._device = value
    
    @property
    def model(self):
        """The model used to generate vectors."""
        return self._model
    
    @model.setter
    def model(self, value):
        """The model used to generate vectors."""
        self._model = value

    @property
    def data(self):
        """Data stored in the database."""
        return self._data
    
    @data.setter
    def data(self, *args):
        """Data stored in the database."""
        if len(args) == 1:
            key = None
            value = args[0]
        elif len(args) == 2:
            key = args[0]
            value = args[1]
        else:
            raise TypeError("'data' must be a dict, list, or list of tuples.")
        

        if self._data is None:
            self._data = Data()

        try:
            if key is not None:
                self._data[key] = value
                return

            if isinstance(value, Data):
                self._data = value
            elif isinstance(value, dict):
                self._data = Data(data=value)
            elif isinstance(value, List):
                self._data = Data()
                for item in value:
                    self._data.append(item)
            elif isinstance(value, List) and all(isinstance(item, Tuple) for item in value):
                for item in value:
                    self._data[item[0]] = item[1]
            else:
                #TODO: Figure out why this is happening
                print('huh?', type(value), type(key))
                raise TypeError
        except TypeError as e:
            traceback.print_exc()
            raise TypeError("'data' must be a dict, list, or list of tuples.")
        except Exception as e:
            raise Exception(f"An unknown error occurred while adding data: {e}")
    
    @property
    def metadata(self):
        """Metadata stored in the database."""
        return self._metadata
    
    @metadata.setter
    def metadata(self, value):
        """Metadata stored in the database."""
        self._metadata = value

    @property
    def vectors(self):
        """Embedding vectors stored in the database."""
        return self._vectors
    
    @vectors.setter
    def vectors(self, value):
        """Embedding vectors stored in the database."""
        self._vectors = value
    
    @property
    def entry_count(self):
        """The number of entries in the database."""
        return len(self._data)
    
    @property
    def info(self):
        """Information about the database."""
        return self._info
    
    @info.setter
    def info(self, value:dict = None):
        """Information about the database."""
        if value is None:
            value = {}
        self._info = value


"""
---------------------
--- Utility Code ----
---------------------
"""
def add_data(data, db: VLite, metadata = None, key = None):
    """Add entry to the database"""
    if key is None:
        key = uuid.uuid4()
    key = str(key)
    
    db.data[key] = data

    if metadata is None:
        metadata = {}
    metadata["id"] = key
    db.metadata[key] = metadata