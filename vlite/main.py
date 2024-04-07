import numpy as np
from uuid import uuid4
from .model import EmbeddingModel
from .utils import chop_and_chunk
import datetime
from .omom import Omom

class VLite:
    def __init__(self, collection=None, device='cpu', model_name='mixedbread-ai/mxbai-embed-large-v1'):
        if collection is None:
            current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            collection = f"vlite_{current_datetime}"
        self.collection = f"{collection}"
        self.device = device
        self.model = EmbeddingModel(model_name) if model_name else EmbeddingModel()
        
        self.omom = Omom()
        self.index = {}

        try:
            with self.omom.read(collection) as omom_file:
                self.index = {
                    chunk_id: {
                        'text': chunk_data['text'],
                        'metadata': chunk_data['metadata'],
                        'binary_vector': np.array(chunk_data['binary_vector'])
                    }
                    for chunk_id, chunk_data in omom_file.metadata.items()
                }
        except FileNotFoundError:
            print(f"Collection file {self.collection} not found. Initializing empty attributes.")

    def add(self, data, metadata=None, item_id=None, need_chunks=True, fast=True):
        data = [data] if not isinstance(data, list) else data
        results = []
        all_chunks = []
        all_metadata = []
        all_ids = []

        for item in data:
            if isinstance(item, dict):
                text_content = item['text']
                item_metadata = item.get('metadata', {})
            else:
                text_content = item
                item_metadata = {}

            if item_id is None:
                item_id = str(uuid4())

            item_metadata.update(metadata or {})

            if need_chunks:
                chunks = chop_and_chunk(text_content, fast=fast)
            else:
                chunks = [text_content]
                print("Encoding text... not chunking")

            all_chunks.extend(chunks)
            all_metadata.extend([item_metadata] * len(chunks))
            all_ids.extend([item_id] * len(chunks))

        encoded_data = self.model.embed(all_chunks, device=self.device)
        binary_encoded_data = self.model.quantize(encoded_data, precision="binary")

        for idx, (chunk, binary_vector, metadata) in enumerate(zip(all_chunks, binary_encoded_data, all_metadata)):
            chunk_id = f"{item_id}_{idx}"
            self.index[chunk_id] = {
                'text': chunk,
                'metadata': metadata,
                'binary_vector': binary_vector.tolist()
            }

        if item_id not in [result[0] for result in results]:
            results.append((item_id, binary_encoded_data, metadata))

        self.save()
        print("Text added successfully.")
        return results

    def retrieve(self, text=None, top_k=5, metadata=None):
        print("Retrieving similar texts...")
        if text:
            print(f"Retrieving top {top_k} similar texts for query: {text}")
            query_chunks = chop_and_chunk(text, fast=True)
            query_vectors = self.model.embed(query_chunks, device=self.device)
            query_binary_vectors = self.model.quantize(query_vectors, precision="binary")

            results = []
            for query_binary_vector in query_binary_vectors:
                chunk_results = self.search(query_binary_vector, top_k, metadata)
                results.extend(chunk_results)

            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:top_k]

            print("Retrieval completed.")
            return [(self.index[idx]['text'], score, self.index[idx]['metadata']) for idx, score in results]
        
    def search(self, query_binary_vector, top_k, metadata=None):
        # Reshape query_binary_vector to 1D array
        query_binary_vector = query_binary_vector.reshape(-1)

        # Perform binary search
        binary_vectors = np.array([item['binary_vector'] for item in self.index.values()])
        binary_similarities = np.einsum('i,ji->j', query_binary_vector, binary_vectors)
        top_k_indices = np.argpartition(binary_similarities, -top_k)[-top_k:]
        top_k_ids = [list(self.index.keys())[idx] for idx in top_k_indices]

        # Apply metadata filter on the retrieved top_k items
        if metadata:
            filtered_ids = []
            for chunk_id in top_k_ids:
                item_metadata = self.index[chunk_id]['metadata']
                if all(item_metadata.get(key) == value for key, value in metadata.items()):
                    filtered_ids.append(chunk_id)
            top_k_ids = filtered_ids[:top_k]

        # Get the similarity scores for the top_k items
        top_k_scores = binary_similarities[top_k_indices]

        return list(zip(top_k_ids, top_k_scores))


    def update(self, id, text=None, metadata=None, vector=None):
        chunk_ids = [chunk_id for chunk_id in self.index if chunk_id.startswith(f"{id}_")]
        if chunk_ids:
            for chunk_id in chunk_ids:
                if text is not None:
                    self.index[chunk_id]['text'] = text
                if metadata is not None:
                    self.index[chunk_id]['metadata'].update(metadata)
                if vector is not None:
                    self.index[chunk_id]['vector'] = vector
            self.save()
            print(f"Item with ID '{id}' updated successfully.")
            return True
        else:
            print(f"Item with ID '{id}' not found.")
            return False


    def delete(self, ids):
        if isinstance(ids, str):
            ids = [ids]

        deleted_count = 0
        for id in ids:
            chunk_ids = [chunk_id for chunk_id in self.index if chunk_id.startswith(f"{id}_")]
            for chunk_id in chunk_ids:
                if chunk_id in self.index:
                    del self.index[chunk_id]
                    deleted_count += 1

        if deleted_count > 0:
            self.save()
            print(f"Deleted {deleted_count} item(s) from the collection.")
        else:
            print("No items found with the specified IDs.")

        return deleted_count

    

    def get(self, ids=None, where=None):
        if ids is not None:
            if isinstance(ids, str):
                ids = [ids]
            items = []
            for id in ids:
                item_chunks = []
                item_metadata = {}
                for chunk_id, chunk_data in self.index.items():
                    if chunk_id.startswith(f"{id}_"):
                        item_chunks.append(chunk_data['text'])
                        item_metadata.update(chunk_data['metadata'])
                if item_chunks:
                    item_text = ' '.join(item_chunks)
                    items.append((item_text, item_metadata))
        else:
            items = []
            item_dict = {}
            for chunk_id, chunk_data in self.index.items():
                item_id = chunk_id.split('_')[0]
                if item_id not in item_dict:
                    item_dict[item_id] = {'chunks': [], 'metadata': {}}
                item_dict[item_id]['chunks'].append(chunk_data['text'])
                item_dict[item_id]['metadata'].update(chunk_data['metadata'])
            for item_id, item_data in item_dict.items():
                item_text = ' '.join(item_data['chunks'])
                item_metadata = item_data['metadata']
                items.append((item_text, item_metadata))

        if where is not None:
            items = [item for item in items if all(item[1].get(key) == value for key, value in where.items())]

        return items

    def set(self, id, text=None, metadata=None, vector=None):
        print(f"Setting attributes for item with ID: {id}")
        chunk_ids = [chunk_id for chunk_id in self.index if chunk_id.startswith(f"{id}_")]
        if chunk_ids:
            self.update(id, text, metadata, vector)
        else:
            self.add(text, metadata=metadata, item_id=id)
            print(f"Item with ID '{id}' created successfully.")
            
                

    def count(self):
        return len(self.index)
    

    def save(self):
        print(f"Saving collection to {self.collection}")
        with self.omom.create(self.collection) as omom_file:
            omom_file.set_header(
                embedding_model=self.model.model_metadata['general.name'],
                embedding_size=self.model.model_metadata.get('bert.embedding_length', 1024),
                embedding_dtype=self.model.embedding_dtype,
                context_length=self.model.model_metadata.get('bert.context_length', 512)
            )
            for chunk_id, chunk_data in self.index.items():
                omom_file.add_embedding(chunk_data['binary_vector'])
                omom_file.add_context(chunk_data['text'])
                omom_file.add_metadata(chunk_id, chunk_data['metadata'])
        print("Collection saved successfully.")

    def clear(self):
        print("Clearing the collection...")
        self.index = {}
        self.omom.delete(self.collection)
        print("Collection cleared.")
    
    def info(self):
        print("Collection Information:")
        print(f"  Items: {self.count()}")
        print(f"  Collection file: {self.collection}")
        print(f"  Embedding model: {self.model}")

    def __repr__(self):
        return f"VLite(collection={self.collection}, device={self.device}, model={self.model})"

    def dump(self):
        return self.index