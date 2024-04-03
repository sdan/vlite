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
        self.collection = f"{collection}.npz"
        self.device = device
        self.model = EmbeddingModel(model_name) if model_name else EmbeddingModel()

        try:
            with np.load(self.collection, allow_pickle=True) as data:
                index_data = data['index'].item()
                self.index = {
                    chunk_id: {
                        'text': chunk_data['text'],
                        'metadata': chunk_data['metadata'],
                        'vector': np.array(chunk_data['vector']),  # Convert back to numpy array
                        'binary_vector': np.array(chunk_data['binary_vector']),  # Convert back to numpy array
                        'int8_vector': np.array(chunk_data['int8_vector'])  # Convert back to numpy array
                    }
                    for chunk_id, chunk_data in index_data.items()
                }
        except FileNotFoundError:
            print(f"Collection file {self.collection} not found. Initializing empty attributes.")
            self.index = {}

    def add(self, data, metadata=None, need_chunks=True, newEmbedding=False):
        """
        Adds text or a list of texts to the collection with optional ID within metadata.

        Args:
            data (str, dict, or list): Text data to be added. Can be a string, a dictionary containing text, id, and/or metadata, or a list of strings or dictionaries.
            metadata (dict, optional): Additional metadata to be appended to each text entry.
            need_chunks (bool, optional): Whether to split the text into chunks before embedding. Defaults to True.

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

            if need_chunks:
                chunks = chop_and_chunk(text_content)
                encoded_data = self.model.embed(chunks, device=self.device)
            else:
                chunks = [text_content]
                print("Encoding text... not chunking")
                encoded_data = self.model.embed(chunks, device=self.device)

            # Quantize the embeddings to binary and int8
            binary_encoded_data = self.model.quantize(encoded_data, precision="binary")
            int8_encoded_data = self.model.quantize(encoded_data, precision="int8")

            for idx, (chunk, vector, binary_vector, int8_vector) in enumerate(zip(chunks, encoded_data, binary_encoded_data, int8_encoded_data)):
                chunk_id = f"{item_id}_{idx}"
                self.index[chunk_id] = {
                    'text': chunk,
                    'metadata': item_metadata,
                    'vector': vector,
                    'binary_vector': binary_vector.tolist(),  # Convert to list for JSON serialization
                    'int8_vector': int8_vector.tolist()  # Convert to list for JSON serialization
                }


            results.append((item_id, encoded_data, item_metadata))

        self.save()
        print("Text added successfully.")
        return results

    def retrieve(self, text=None, top_k=5, metadata=None, newEmbedding=False):
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
            query_binary_vector = self.model.quantize(query_vector, precision="binary")
            query_int8_vector = self.model.quantize(query_vector, precision="int8")

            # Perform binary search and rescoring
            results = self.retrieval_rescore(query_binary_vector, query_int8_vector, top_k, metadata)

            print("Retrieval completed.")
            return [(self.index[idx]['text'], score, self.index[idx]['metadata']) for idx, score in results]

    def retrieval_rescore(self, query_binary_vector, query_int8_vector, top_k, metadata=None):
        """
        Performs retrieval using binary search and rescoring using int8 embeddings.

        Args:
            query_binary_vector (numpy.ndarray): Binary vector of the query.
            query_int8_vector (numpy.ndarray): Int8 vector of the query.
            top_k (int): Number of top similar texts to retrieve.
            metadata (dict, optional): Metadata to filter the retrieved texts.

        Returns:
            list: A list of tuples containing the chunk IDs and their similarity scores.
        """
        # Perform binary search
        binary_vectors = np.array([item['binary_vector'] for item in self.index.values()])
        similarities = np.dot(query_binary_vector, binary_vectors.T).flatten()

        # Apply metadata filter while finding similar texts
        if metadata:
            filtered_indices = []
            for idx, item_id in enumerate(self.index.keys()):  # Iterate over item IDs
                item_metadata = self.index[item_id]['metadata']
                if all(item_metadata.get(key) == value for key, value in metadata.items()):
                    filtered_indices.append(idx)
            if len(filtered_indices) == top_k:  # Stop when we have found top_k
                top_k_ids = [list(self.index.keys())[idx] for idx in filtered_indices]
            else:
                top_k_ids = [list(self.index.keys())[idx] for idx in np.argsort(similarities)[-top_k:][::-1]]
        else:
            top_k_ids = [list(self.index.keys())[idx] for idx in np.argsort(similarities)[-top_k:][::-1]]

        # Perform rescoring using int8 embeddings
        int8_vectors = np.array([self.index[idx]['int8_vector'] for idx in top_k_ids])
        rescored_similarities = self.model.rescore(query_int8_vector, int8_vectors)

        # Sort the results based on the rescored similarities
        sorted_indices = np.argsort(rescored_similarities)[::-1]
        sorted_ids = [top_k_ids[idx] for idx in sorted_indices]
        sorted_scores = rescored_similarities[sorted_indices]

        return list(zip(sorted_ids, sorted_scores))
        
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
        for id in ids:
            if id in self.index:
                del self.index[id]
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
        if id in self.index:
            if text is not None:
                self.index[id]['text'] = text

            if metadata is not None:
                self.index[id]['metadata'].update(metadata)

            if vector is not None:
                self.index[id]['vector'] = vector

            self.save()
            print(f"Item with ID '{id}' updated successfully.")
            return True
        else:
            print(f"Item with ID '{id}' not found.")
            return False
    
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
            items = [(self.index[id]['text'], self.index[id]['metadata']) for id in self.index if id in id_set]
        else:
            items = [(self.index[id]['text'], self.index[id]['metadata']) for id in self.index]

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
        if id in self.index:
            if text is not None:
                self.index[id]['text'] = text
            if metadata is not None:
                self.index[id]['metadata'].update(metadata)
            if vector is not None:
                self.index[id]['vector'] = vector
            self.save()
        else:
            print(f"Item with ID {id} not found.")

    def count(self):
        """
        Returns the number of items in the collection.

        Returns:
            int: The count of items in the collection.
        """
        return len(self.index)

    def save(self):
        """
        Saves the current state of the collection to a file.
        """
        print(f"Saving collection to {self.collection}")
        index_data = {
            chunk_id: {
                'text': chunk_data['text'],
                'metadata': chunk_data['metadata'],
                'vector': chunk_data['vector'],
                'binary_vector': chunk_data['binary_vector'],
                'int8_vector': chunk_data['int8_vector']
            }
            for chunk_id, chunk_data in self.index.items()
        }
        with open(self.collection, 'wb') as f:
            np.savez(f, index=index_data)
        print("Collection saved successfully.")
        

    def clear(self):
        """
        Clears the entire collection, removing all items and resetting the attributes.
        """
        print("Clearing the collection...")
        self.index = {}
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
        return self.index