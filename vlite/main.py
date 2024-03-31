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


    def add(self, data, metadata=None):
        """
        Adds text or a list of texts to the collection with optional ID and metadata.

        Args:
            data (str, dict, or list): Text data to be added. Can be a string, a dictionary
                containing text, id, and/or metadata, or a list of strings or dictionaries.
            metadata (dict, optional): Additional metadata to be appended to each text entry.

        Returns:
            list: A list of tuples, each containing the ID of the added text and the updated vectors array.
        """
        print("Adding text to the collection...")

        data = [data] if not isinstance(data, list) else data

        results = []
        for item in data:
            text_content, id, item_metadata = (
                (item['text'], item.get('id', str(uuid4())), item.get('metadata', {}))
                if isinstance(item, dict)
                else (item, str(uuid4()), {})
            )

            item_metadata.update(metadata or {})

            chunks = chop_and_chunk(text_content)
            encoded_data = self.model.embed(chunks, device=self.device)
            self.vectors = np.vstack((self.vectors, encoded_data))

            update_metadata = lambda idx: {
                **self.metadata.get(idx, {}),
                **item_metadata,
                'index': id
            }
            self.metadata.update({idx: update_metadata(idx) for idx in range(len(self.texts), len(self.texts) + len(chunks))})

            self.texts.extend(chunks)
            results.append((id, self.vectors))

        self.save()
        print("Text added successfully.")
        return results

    def retrieve(self, text=None, id=None, top_k=5, metadata=None):
        """
        Retrieves similar texts from the collection based on text content, ID, or metadata.

        Args:
            text (str, optional): Query text for finding similar texts.
            id (str, optional): ID of the text to retrieve.
            top_k (int, optional): Number of top similar texts to retrieve. Defaults to 5.
            metadata (dict, optional): Metadata to filter the retrieved texts.

        Returns:
            tuple: A tuple containing a list of similar texts, their similarity scores, and metadata (if applicable).
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

            # Filter by metadata if provided
            if metadata:
                filtered_indices = []
                for idx in top_k_idx:
                    item_metadata = self.metadata.get(idx, {})
                    if all(item_metadata.get(key) == value for key, value in metadata.items()):
                        filtered_indices.append(idx)
                top_k_idx = filtered_indices

            print("Retrieval completed.")
            return [self.texts[idx] for idx in top_k_idx], [similarities[idx] for idx in top_k_idx], [self.metadata[idx] for idx in top_k_idx]
            
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


    def set(self, id, text=None, metadata=None, vector=None):
        """
        Updates the attributes of an item in the collection by ID.

        Args:
            id (str): ID of the item to update.
            text (str, optional): Updated text content of the item.
            metadata (dict, optional): Updated metadata of the item.
            vector (numpy.ndarray, optional): Updated embedding vector of the item.
        """
        print(f"Setting attributes for item with ID: {id}")
        idx = next((i for i, x in enumerate(self.metadata) if x.get('index') == id), None)
        if idx is not None:
            if text is not None:
                self.texts[idx] = text
            if metadata is not None:
                self.metadata[idx].update(metadata)
            if vector is not None:
                self.vectors[idx] = vector
            self.save()
        else:
            print(f"Item with ID {id} not found.")

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

    def clear(self):
        """
        Clears the entire collection, removing all items and resetting the attributes.
        """
        print("Clearing the collection...")
        self.texts = []
        self.metadata = {}
        self.vectors = np.empty((0, self.model.dimension))
        self.save()
        print("Collection cleared.")
    
    def info(self):
        """
        Prints information about the collection, including the number of items, collection file path,
        and the embedding model used.
        """
        print("Collection Information:")
        print(f"  Items: {self.count()}")
        print(f"  Collection file: {self.collection}")
        print(f"  Embedding model: {self.model}")