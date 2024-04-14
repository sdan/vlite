import numpy as np
from uuid import uuid4
from .utils import check_cuda_available, check_mps_available 
from .model import EmbeddingModel
from .utils import chop_and_chunk
import datetime
from .ctx import Ctx
import time

class VLite:
    def __init__(self, collection=None, device=None, model_name='mixedbread-ai/mxbai-embed-large-v1'):
        start_time = time.time()
        
        if device is None:
            if check_cuda_available():
                device = 'cuda'
            elif check_mps_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        print(f"Initializing VLite with device: {device}")
        self.device = device
        
        if collection is None:
            current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            collection = f"vlite_{current_datetime}"
        self.collection = f"{collection}"
        self.model = EmbeddingModel(model_name) if model_name else EmbeddingModel()
        
        self.ctx = Ctx()
        self.index = {}

        try:
            ctx_file = self.ctx.read(collection)
            ctx_file.load()
            print("Number of embeddings: ", len(ctx_file.embeddings))
            print("Number of metadata: ", len(ctx_file.metadata))
            self.index = {
                chunk_id: {
                    'text': ctx_file.contexts[idx] if idx < len(ctx_file.contexts) else "",
                    'metadata': ctx_file.metadata.get(chunk_id, {}),
                    'binary_vector': np.array(ctx_file.embeddings[idx]) if idx < len(ctx_file.embeddings) else np.zeros(self.model.embedding_size)
                }
                for idx, chunk_id in enumerate(ctx_file.metadata.keys())
            }
        except FileNotFoundError:
            print(f"Collection file {self.collection} not found. Initializing empty attributes.")
        
        end_time = time.time()
        print(f"[__init__] Execution time: {end_time - start_time:.5f} seconds")
        print(f"Using device: {self.device}")

    def add(self, data, metadata=None, item_id=None, need_chunks=True, fast=True):
        start_time = time.time()
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
        end_time = time.time()
        print(f"[add] Execution time: {end_time - start_time:.5f} seconds")
        return results

    def retrieve(self, text=None, top_k=5, metadata=None, return_scores=False):
        start_time = time.time()
        print("Retrieving similar texts...")
        if text:
            print(f"Retrieving top {top_k} similar texts for query: {text}")
                        
            query_vectors = self.model.embed(text, device=self.device)
            
            query_binary_vectors = self.model.quantize(query_vectors, precision="binary")
            
            # Perform search on the query binary vectors
            results = []
            for query_binary_vector in query_binary_vectors:
                chunk_results = self.rank_and_filter(query_binary_vector, top_k, metadata)
                results.extend(chunk_results)
            
            # Sort the results by similarity score 
            results.sort(key=lambda x: x[1])
            results = results[:top_k]
            
            print("Retrieval completed.")
            end_time = time.time()
            print(f"[retrieve] Execution time: {end_time - start_time:.5f} seconds")
            if return_scores:
                return [(idx, self.index[idx]['text'], self.index[idx]['metadata'], score) for idx, score in results]
            else:
                return [(idx, self.index[idx]['text'], self.index[idx]['metadata']) for idx, _ in results]
        
    def rank_and_filter(self, query_binary_vector, top_k, metadata=None):
        start_time = time.time()
        query_binary_vector = np.array(query_binary_vector).reshape(-1)
        
        corpus_binary_vectors = np.array([item['binary_vector'] for item in self.index.values()])
        top_k_indices, top_k_scores = self.model.search(query_binary_vector, corpus_binary_vectors, top_k)
        
        print(f"Top {top_k} indices: {top_k_indices}")
        print(f"Top {top_k} scores: {top_k_scores}")
        print(f"No. of items in the collection: {len(self.index)}")
        print(f"Vlite count: {self.count()}")
        
        top_k_ids = [list(self.index.keys())[idx] for idx in top_k_indices]

        # Apply metadata filter on the retrieved top_k items
        if metadata:
            filtered_ids = []
            for chunk_id in top_k_ids:
                item_metadata = self.index[chunk_id]['metadata']
                if all(item_metadata.get(key) == value for key, value in metadata.items()):
                    filtered_ids.append(chunk_id)
            top_k_ids = filtered_ids[:top_k]
            top_k_scores = top_k_scores[:len(top_k_ids)]
            
        end_time = time.time()
        print(f"[rank_and_filter] Execution time: {end_time - start_time:.5f} seconds")

        return list(zip(top_k_ids, top_k_scores))
        


    def update(self, id, text=None, metadata=None, vector=None):
        start_time = time.time()
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
            end_time = time.time()
            print(f"[update] Execution time: {end_time - start_time:.5f} seconds")
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