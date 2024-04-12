import numpy as np
from uuid import uuid4
from .model import EmbeddingModel
from .utils import chop_and_chunk
import datetime
from .ctx import Ctx

class VLite:
    def __init__(self, collection=None, device='cpu', model_name='mixedbread-ai/mxbai-embed-large-v1'):
        if collection is None:
            current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            collection = f"vlite_{current_datetime}"
        self.collection = f"{collection}"
        self.device = device
        self.model = EmbeddingModel(model_name) if model_name else EmbeddingModel()
        
        self.ctx = Ctx()
        self.index = {}

        try:
            ctx_file = self.ctx.read(collection)
            ctx_file.load()
            # debug print
            print("Number of embeddings: ", len(ctx_file.embeddings))
            print("Number of metadata: ", len(ctx_file.metadata))
            self.index = {
                chunk_id: {
                    'text': ctx_file.contexts[idx] if idx < len(ctx_file.contexts) else "",
                    'metadata': ctx_file.metadata.get(chunk_id, {}),
                    'binary_vector': np.frombuffer(ctx_file.embeddings[idx], dtype=np.uint8) if idx < len(ctx_file.embeddings) else np.zeros(self.model.embedding_size // 8, dtype=np.uint8),
                    'int8_vector': np.frombuffer(ctx_file.int8_embeddings[idx], dtype=np.int8) if idx < len(ctx_file.int8_embeddings) else np.zeros(self.model.embedding_size, dtype=np.int8)
                }
                for idx, chunk_id in enumerate(ctx_file.metadata.keys())
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
        int8_encoded_data = self.model.quantize(encoded_data, precision="int8")
        for idx, (chunk, binary_vector, int8_vector, metadata) in enumerate(zip(all_chunks, binary_encoded_data, int8_encoded_data, all_metadata)):
            chunk_id = f"{item_id}_{idx}"
            self.index[chunk_id] = {
                'text': chunk,
                'metadata': metadata,
                'binary_vector': binary_vector.tobytes(),
                'int8_vector': int8_vector.tobytes()
            }

        if item_id not in [result[0] for result in results]:
            results.append((item_id, binary_encoded_data, metadata))

        self.save()
        print("Text added successfully.")
        return results

    def retrieve(self, text, top_k=5, metadata=None, return_scores=False, rescore_top_k=40):
        # Embed the query text
        query_embedding = self.model.embed([text])[0]

        # Quantize the query embedding to binary
        binary_query_embedding = self.model.quantize([query_embedding], precision="binary")[0]

        # Perform binary search to retrieve top_k * rescore_top_k candidates
        candidate_ids, candidate_scores = self._binary_search(binary_query_embedding, top_k * rescore_top_k)

        # Rescore the candidates using int8 embeddings
        rescored_ids, rescored_scores = self._rescore_candidates(query_embedding, candidate_ids, rescore_top_k)

        # Filter results based on metadata if provided
        if metadata:
            filtered_ids = [idx for idx in rescored_ids if all(self.index[idx]['metadata'].get(k) == v for k, v in metadata.items())][:top_k]
        else:
            filtered_ids = rescored_ids[:top_k]

        # Retrieve the text and metadata for the filtered results
        results = [(idx, self.index[idx]['text'], self.index[idx]['metadata']) for idx in filtered_ids]

        if return_scores:
            scores = [rescored_scores[rescored_ids.index(idx)] for idx in filtered_ids]
            results = [(idx, text, metadata, score) for (idx, text, metadata), score in zip(results, scores)]

        return results

    def _binary_search(self, query_embedding, top_k):
        # Retrieve all binary vectors from the index
        binary_vectors = np.stack([np.frombuffer(chunk_data['binary_vector'], dtype=np.uint8).reshape(-1) for chunk_data in self.index.values()])
        
        # Compute Hamming distances between the query embedding and all binary vectors
        distances = np.sum(query_embedding != binary_vectors, axis=1)
        
        # Get the top_k indices and scores (Hamming distances)
        num_distances = len(distances)
        if top_k >= num_distances:
            top_k_indices = np.arange(num_distances)
        else:
            top_k_indices = np.argpartition(distances, top_k)[:top_k]
        top_k_scores = distances[top_k_indices]
        
        # Map the indices to chunk IDs
        top_k_ids = [list(self.index.keys())[idx] for idx in top_k_indices]
        
        return top_k_ids, top_k_scores

    def _rescore_candidates(self, query_embedding, candidate_ids, top_k):
        # Retrieve the int8 embeddings for the candidates
        candidate_embeddings = np.stack([np.frombuffer(self.index[idx]['int8_vector'], dtype=np.int8) for idx in candidate_ids])
        
        # Compute the dot product between the query embedding and candidate int8 embeddings
        scores = np.dot(query_embedding, candidate_embeddings.T)
        
        # Get the top_k indices and scores
        top_k_indices = np.argsort(-scores)[:top_k]
        top_k_scores = scores[top_k_indices]
        
        # Map the indices to chunk IDs
        top_k_ids = [candidate_ids[idx] for idx in top_k_indices]
        
        return top_k_ids, top_k_scores
    
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
                    items.append((id, item_text, item_metadata))
        else:
            items = []
            for chunk_id, chunk_data in self.index.items():
                item_id = chunk_id.split('_')[0]
                item_text = chunk_data['text']
                item_metadata = chunk_data['metadata']
                items.append((item_id, item_text, item_metadata))

        if where is not None:
            items = [item for item in items if all(item[2].get(key) == value for key, value in where.items())]

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
        with self.ctx.create(self.collection) as ctx_file:
            ctx_file.set_header(
                embedding_model="mixedbread-ai/mxbai-embed-large-v1",
                embedding_size=self.model.model_metadata.get('bert.embedding_length', 1024),
                embedding_dtype=self.model.embedding_dtype,
                context_length=self.model.model_metadata.get('bert.context_length', 512)
            )
            for chunk_id, chunk_data in self.index.items():
                ctx_file.add_embedding(chunk_data['binary_vector'])
                ctx_file.add_context(chunk_data['text'])
                if 'metadata' in chunk_data:
                    ctx_file.add_metadata(chunk_id, chunk_data['metadata'])
        print("Collection saved successfully.")

    def clear(self):
        print("Clearing the collection...")
        self.index = {}
        self.ctx.delete(self.collection)
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