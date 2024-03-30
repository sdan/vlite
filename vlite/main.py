import numpy as np
from uuid import uuid4
from .model import EmbeddingModel
from .utils import chop_and_chunk
import datetime

class VLite:
    """
    A simple vector database for text embedding and retrieval.

    Attributes:
        collection (str): Path to the collection file.
        device (str): Device to use for embedding ('cpu' or 'cuda').
        model (EmbeddingModel): The embedding model used for text representation.

    Methods:
        add(text, id=None, metadata=None): Adds a text to the collection with optional ID and metadata.
        retrieve(text=None, id=None, top_k=5): Retrieves similar texts from the collection.
        save(): Saves the collection to a file.
    """
    def __init__(self, collection=None, device='cpu', model_name='mixedbread-ai/mxbai-embed-large-v1'):
        if collection is None:
            current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            collection = f"vlite_{current_datetime}.npz"
        self.collection = collection
        self.device = device
        self.model = EmbeddingModel(model_name) if model_name else EmbeddingModel()

        # Load existing collection if available, otherwise initialize empty attributes
        try:
            with np.load(self.collection, allow_pickle=True) as data:
                self.texts = data['texts'].tolist()
                self.metadata = data['metadata'].tolist()
                self.vectors = data['vectors']
        except FileNotFoundError:
            print(f"Collection file {self.collection} not found. Initializing empty attributes.")
            self.texts = []  # List to store text chunks
            self.metadata = {}  # Dictionary to store metadata
            self.vectors = np.empty((0, self.model.dimension))  # Empty array to store embedding vectors

    def add(self, text, id=None, metadata=None):
        """
        Adds text to the collection with optional ID and metadata.

        Args:
            text (str or dict): Text to be added. If a dictionary is provided,
                it should contain the text, id, and metadata.
            id (str, optional): Unique identifier for the text. Defaults to a UUID.
            metadata (dict, optional): Metadata associated with the text.

        Returns:
            tuple: A tuple containing the ID of the added text and the updated vectors array.
        """
        print("Adding text to the collection...")

        if isinstance(text, dict):
            id = text.get('id', str(uuid4()))
            text_content = text.get('text')
            metadata = text.get('metadata', {})
        else:
            id = id or str(uuid4())
            text_content = text

        chunks = chop_and_chunk(text_content)
        encoded_data = self.model.embed(chunks, device=self.device)
        self.vectors = np.vstack((self.vectors, encoded_data))

        for chunk in chunks:
            self.texts.append(chunk)
            idx = len(self.texts) - 1
            self.metadata[idx] = metadata
            self.metadata[idx]['index'] = id

        self.save()
        print("Text added successfully.")

        return id, self.vectors

    def retrieve(self, text=None, id=None, top_k=5):
        """
        Retrieves similar texts from the collection based on text content or ID.

        Args:
            text (str, optional): Query text for finding similar texts.
            id (str, optional): ID of the text to retrieve.
            top_k (int, optional): Number of top similar texts to retrieve. Defaults to 5.

        Returns:
            tuple: A tuple containing a list of similar texts and their similarity scores.
        """
        print("Retrieving similar texts...")
        if id:
            print(f"Retrieving text with ID: {id}")
            return self.metadata[id]
        if text:
            print(f"Retrieving top {top_k} similar texts for query: {text}")
            query_vector = self.model.embed([text], device=self.device)
            similarities = np.dot(query_vector, self.vectors.T).flatten()
            top_k_idx = np.argsort(similarities)[-top_k:][::-1]
            print("Retrieval completed.")
            return [self.texts[idx] for idx in top_k_idx], similarities[top_k_idx]

    def delete(self, id):
        """
        Deletes a text from the collection by ID.
        """
        print(f"Deleting text with ID: {id}")
        del self.texts[id]
        del self.metadata[id]
        self.vectors = np.delete(self.vectors, id, axis=0)
        self.save()
    
    def update(self, id, text, metadata=None):
        """
        Updates a text in the collection by ID.
        """
        print(f"Updating text with ID: {id}")
        self.delete(id)
        self.add(text, id, metadata)
    
    def get(self, ids=None, where=None):
        """
        Retrieves items from the collection based on IDs or metadata.

        Args:
            ids (list, optional): List of IDs to retrieve. If not provided, all items will be returned.
            where (dict, optional): Metadata filter to apply. Items matching the filter will be returned.

        Returns:
            list: A list of retrieved items.
        """
        if ids is None:
            ids = range(len(self.texts))

        if where is None:
            return [(self.texts[idx], self.metadata[idx]) for idx in ids if idx in self.metadata]
        else:
            return [(self.texts[idx], self.metadata[idx]) for idx in ids if idx in self.metadata and all(self.metadata[idx].get(k) == v for k, v in where.items())]
            
    def count(self):
        """
        Returns the number of items in the collection.

        Returns:
            int: The count of items in the collection.
        """
        return len(self.texts)

    def save(self):
        """
        Saves the current state of the collection to a file.
        """
        print(f"Saving collection to {self.collection}")
        with open(self.collection, 'wb') as f:
            np.savez(f, texts=self.texts, metadata=self.metadata, vectors=self.vectors)
        print("Collection saved successfully.")