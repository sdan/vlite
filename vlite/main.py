import numpy as np
from uuid import uuid4
from .utils import check_cuda_available, check_mps_available
from .model import EmbeddingModel
from .utils import chop_and_chunk
from .index import BinaryVectorIndex
import datetime
from .ctx import Ctx
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)    

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
        print(f"[VLite.__init__] Initializing VLite with device: {device}")
        self.device = device
        if collection is None:
            current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            collection = f"vlite_{current_datetime}"
        self.collection = f"{collection}"
        self.model = EmbeddingModel(model_name, device=device) if model_name else EmbeddingModel()
        self.ctx = Ctx()
        self.index = {}
        self.binary_index = BinaryVectorIndex()
        try:
            ctx_file = self.ctx.read(collection)
            ctx_file.load()
            print(f"[VLite.__init__] Number of embeddings: {len(ctx_file.embeddings)}")
            print(f"[VLite.__init__] Number of metadata: {len(ctx_file.metadata)}")

            chunk_ids = list(ctx_file.metadata.keys())
            self.index = {
                chunk_id: {
                    'text': ctx_file.contexts[idx] if idx < len(ctx_file.contexts) else "",
                    'metadata': ctx_file.metadata.get(chunk_id, {}),
                }
                for idx, chunk_id in enumerate(chunk_ids)
            }

            self.binary_index.add_batch(
                chunk_ids,
                [np.array(embedding) if idx < len(ctx_file.embeddings) else np.zeros(64) for idx, embedding in enumerate(ctx_file.embeddings)]
            )
        except FileNotFoundError:
            logger.warning(f"[VLite.__init__] Collection file {self.collection} not found. Initializing empty attributes.")
        end_time = time.time()
        print(f"[VLite.__init__] Execution time: {end_time - start_time:.5f} seconds")
        print(f"[VLite.__init__] Using device: {self.device}")
        
    def add(self, data, metadata=None, item_id=None, need_chunks=False, fast=True):
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
            print("[VLite.add] Encoding text... not chunking")
            all_chunks.extend(chunks)
            all_metadata.extend([item_metadata] * len(chunks))
            all_ids.extend([item_id] * len(chunks))
        binary_encoded_data = self.model.embed(all_chunks, precision="binary")
        for idx, (chunk, binary_vector, metadata) in enumerate(zip(all_chunks, binary_encoded_data, all_metadata)):
            chunk_id = f"{all_ids[idx]}_{idx}"
            self.index[chunk_id] = {
                'text': chunk,
                'metadata': metadata,
                'item_id': all_ids[idx]  # Store the item ID along with the chunk data
            }
            self.binary_index.add(chunk_id, binary_vector)
            print(f"[VLite.add] Added chunk ID: {chunk_id}")
            print(f"[VLite.add] Main index keys: {list(self.index.keys())}")
            print(f"[VLite.add] Binary index keys: {list(self.binary_index.index.keys())}")
        if item_id not in [result[0] for result in results]:
            results.append((item_id, binary_encoded_data, metadata))
        self.save()
        print("[VLite.add] Text added successfully.")
        end_time = time.time()
        print(f"[VLite.add] Execution time: {end_time - start_time:.5f} seconds")
        return results

    def retrieve(self, text=None, top_k=5, metadata=None, return_scores=False):
        start_time = time.time()
        if text:
            query_binary_vectors = self.model.embed(text, precision="binary")
            
            top_k_ids, top_k_scores = self.binary_index.search(query_binary_vectors, top_k)
            
            chunk_data = [self.index[chunk_id] for chunk_id in top_k_ids if chunk_id in self.index]
            
            if metadata:
                chunk_data = [data for data in chunk_data if all(data['metadata'].get(key) == value for key, value in metadata.items())]
            
            texts = [data['text'] for data in chunk_data]
            metadatas = [data['metadata'] for data in chunk_data]
            scores = [score for chunk_id, score in zip(top_k_ids, top_k_scores) if chunk_id in self.index][:len(chunk_data)]
            
            if return_scores:
                results = list(zip(chunk_data, texts, metadatas, scores))
            else:
                results = list(zip(chunk_data, texts, metadatas))
            
            end_time = time.time()
            logger.debug(f"[VLite.retrieve] Execution time: {end_time - start_time:.5f} seconds")
            return results
        else:
            logger.warning("[VLite.retrieve] No query text provided.")
            return []

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
                    self.binary_index.remove(chunk_id)
                    self.binary_index.add(chunk_id, vector)
            self.save()
            print(f"[VLite.update] Item with ID '{id}' updated successfully.")
            end_time = time.time()
            print(f"[VLite.update] Execution time: {end_time - start_time:.5f} seconds")
            return True
        else:
            logger.warning(f"[VLite.update] Item with ID '{id}' not found.")
            return False

    def delete(self, ids):
        if isinstance(ids, str):
            ids = [ids]
        deleted_count = 0
        for id in ids:
            chunk_ids = [chunk_id for chunk_id in self.index if chunk_id.startswith(f"{id}_")]
            for chunk_id in chunk_ids:
                if chunk_id in self.index:
                    self.binary_index.remove(chunk_id)
                    del self.index[chunk_id]
                    deleted_count += 1
        if deleted_count > 0:
            self.save()
            print(f"[VLite.delete] Deleted {deleted_count} item(s) from the collection.")
        else:
            logger.warning("[VLite.delete] No items found with the specified IDs.")
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
        print(f"[VLite.set] Setting attributes for item with ID: {id}")
        self.delete(id)  # Remove existing item with the same ID
        self.add(text, metadata=metadata, item_id=id)  # Add the item as a new entry
        print(f"[VLite.set] Item with ID '{id}' created successfully.")

    def count(self):
        return len(self.index)

    def save(self):
        print(f"[VLite.save] Saving collection to {self.collection}")
        with self.ctx.create(self.collection) as ctx_file:
            ctx_file.set_header(
                embedding_model="mixedbread-ai/mxbai-embed-large-v1",
                embedding_size=64,
                embedding_dtype=self.model.embedding_dtype,
                context_length=512
            )
            for chunk_id, chunk_data in self.index.items():
                binary_vector = self.binary_index.index.get(chunk_id, np.zeros(64))
                ctx_file.add_embedding(binary_vector)
                ctx_file.add_context(chunk_data['text'])
                if 'metadata' in chunk_data:
                    ctx_file.add_metadata(chunk_id, chunk_data['metadata'])
        print("[VLite.save] Collection saved successfully.")

    def clear(self):
        print("[VLite.clear] Clearing the collection...")
        self.index = {}
        self.binary_index = BinaryVectorIndex()
        self.ctx.delete(self.collection)
        print("[VLite.clear] Collection cleared.")

    def info(self):
        print("[VLite.info] Collection Information:")
        print(f"[VLite.info] Items: {self.count()}")
        print(f"[VLite.info] Collection file: {self.collection}")
        print(f"[VLite.info] Embedding model: {self.model}")

    def __repr__(self):
        return f"VLite(collection={self.collection}, device={self.device}, model={self.model})"

    def dump(self):
        return {
            'index': self.index,
            'binary_index': self.binary_index.index
        }