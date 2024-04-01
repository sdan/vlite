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
        self.__version__ = '1.1.1'
        if collection is None:
            current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            collection = f"vlite_{current_datetime}.npz"
        self.collection = f"{collection}.npz"
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
        Adds text or a list of texts to the collection with optional ID within metadata.

        Args:
            data (str, dict, or list): Text data to be added. Can be a string, a dictionary containing text, id, and/or metadata, or a list of strings or dictionaries.
            metadata (dict, optional): Additional metadata to be appended to each text entry.

        Returns:
            list: A list of tuples, each containing the ID of the added text and the updated vectors array.
        """
        print("Adding text to the collection...")
        data = [data] if not isinstance(data, list) else data
        results = []

        for item in data:
            if isinstance(item, dict):
                text_content = item['text']
                item_metadata = item.get('metadata', {})
                item_id = item_metadata.get('id', str(uuid4()))
            else:
                text_content = item
                item_metadata = {}
                item_id = str(uuid4())

            item_metadata.update(metadata or {})
            item_metadata['id'] = item_id 
            
            chunks = chop_and_chunk(text_content)
            encoded_data = self.model.embed(chunks, device=self.device)
            self.vectors = np.vstack((self.vectors, encoded_data))

            for idx in range(len(self.texts), len(self.texts) + len(chunks)):
                        self.metadata[idx] = item_metadata

            self.texts.extend(chunks)
            results.append((item_id, self.vectors))
            
        self.save()
        print("Text added successfully.")
        return results

    def retrieve(self, text=None, top_k=5, metadata=None):
        """
        Retrieves similar texts from the collection based on text content, ID, or metadata.

        Args:
            text (str, optional): Query text for finding similar texts.
            top_k (int, optional): Number of top similar texts to retrieve. Defaults to 5.
            metadata (dict, optional): Metadata to filter the retrieved texts.

        Returns:
            tuple: A tuple containing a list of similar texts, their similarity scores, and metadata (if applicable).
        """
        print("Retrieving similar texts...")
        if text:
            print(f"Retrieving top {top_k} similar texts for query: {text}")
            query_vector = self.model.embed([text], device=self.device)
            similarities = np.dot(query_vector, self.vectors.T).flatten()

            # Apply metadata filter while finding similar texts
            if metadata:
                filtered_indices = []
                for idx in np.argsort(similarities)[::-1]:  # Start from the most similar
                    item_metadata = self.metadata.get(idx, {})
                    if all(item_metadata.get(key) == value for key, value in metadata.items()):
                        filtered_indices.append(idx)
                    if len(filtered_indices) == top_k:  # Stop when we have found top_k
                        break
                top_k_idx = filtered_indices
            else:
                top_k_idx = np.argsort(similarities)[-top_k:][::-1]

            print("Retrieval completed.")
            return [(self.texts[idx], similarities[idx], self.metadata[idx]) for idx in top_k_idx]
        
    def delete(self, ids):
        """
        Deletes items from the collection by their IDs.

        Args:
            ids (list or str): A single ID or a list of IDs of the items to delete.

        Returns:
            int: The number of items deleted from the collection.
        """
        if isinstance(ids, str):
            ids = [ids]

        deleted_count = 0
        for idx in range(len(self.metadata) - 1, -1, -1):  # Iterate in reverse order
            if self.metadata[idx].get('id') in ids:
                del self.texts[idx]
                del self.metadata[idx]
                self.vectors = np.delete(self.vectors, idx, axis=0)
                deleted_count += 1

        if deleted_count > 0:
            self.save()
            print(f"Deleted {deleted_count} item(s) from the collection.")
        else:
            print("No items found with the specified IDs.")

        return deleted_count
    
    def update(self, id, text=None, metadata=None, vector=None):
        """
        Updates an item in the collection by its ID.

        Args:
            id (str): The ID of the item to update.
            text (str, optional): The updated text content of the item.
            metadata (dict, optional): The updated metadata of the item.
            vector (numpy.ndarray, optional): The updated embedding vector of the item.

        Returns:
            bool: True if the item was successfully updated, False otherwise.
        """
        print("UPDATERR",self.metadata)
        idx = next((i for i, x in self.metadata.items() if x.get('id') == id), None)
        
        if idx is None:
            print(f"Item with ID '{id}' not found.")
            return False

        if text is not None:
            self.texts[idx] = text

        if metadata is not None:
            self.metadata[idx].update(metadata)

        if vector is not None:
            self.vectors[idx] = vector

        self.save()
        print(f"Item with ID '{id}' updated successfully.")
        return True
    
    def get(self, ids=None, where=None):
        """
        Retrieves items from the collection based on IDs and/or metadata.

        Args:
            ids (list, optional): List of IDs to retrieve. If provided, only items with the specified IDs will be returned.
            where (dict, optional): Metadata filter to apply. Items matching the filter will be returned.

        Returns:
            list: A list of retrieved items, each item being a tuple of (text, metadata).
        """
        if ids is not None:
            # Convert ids to a set for faster membership testing
            id_set = set(ids)
            items = [(self.texts[idx], self.metadata[idx]) for idx in range(len(self.texts)) if self.metadata[idx].get('id') in id_set]
        else:
            items = [(self.texts[idx], self.metadata[idx]) for idx in range(len(self.texts))]

        if where is not None:
            # Filter items based on metadata
            items = [item for item in items if all(item[1].get(key) == value for key, value in where.items())]

        return items


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
        idx = next((i for i, x in self.metadata.items() if x.get('id') == id), None)

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

    def __repr__(self):
        return f"VLite(collection={self.collection}, device={self.device}, model={self.model})"

    def dump(self):
        """
        Dumps the collection data to a dictionary for serialization.

        Returns:
            dict: A dictionary containing the collection data.
        """
        return {
            'texts': self.texts,
            'metadata': self.metadata,
            'vectors': self.vectors
        }