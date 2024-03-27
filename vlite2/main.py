import numpy as np
from .model import EmbeddingModel
from .utils import chop_and_chunk
from usearch.index import Index, Matches
import datetime
import os

class VLite2:
    def __init__(self, vdb_name: str = None, device: str ='mps', embedding_model: str = 'all-MiniLM-L6-v2'):
        if vdb_name is None:
            current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            vdb_name = f"vlite2_{current_datetime}"
        else:
            vdb_name = os.path.splitext(vdb_name)[0]  # remove file extension; don't want this on the vdb name

        self.__name = vdb_name
        self.__metadata_file = f"{vdb_name}.info"
        self.__index_file = f"{vdb_name}.index"

        metadata_exists = os.path.exists(self.__metadata_file)
        index_exists = os.path.exists(self.__index_file)
        if metadata_exists != index_exists:
            raise Exception("Must have BOTH .info and .index file present. Cannot continue unless neither or both files exist.")

        self.device = device
        self.__embed_model = EmbeddingModel(model_name=embedding_model)

        self.__index: Index = Index(ndim=self.__embed_model.dimension, metric='cos', path=self.__index_file)  # existence handled within USearch Index Object

        if metadata_exists:
            with np.load(self.__metadata_file, allow_pickle=True) as data:
                self.__texts = data['texts'].tolist()
                self.__metadata = data['metadata'].tolist()
                self.__chunk_id = int(data['chunk_id'])
                self.__document_id = int(data['document_id'])
        else:
            self.__texts = {}
            self.__metadata = {}
            self.__chunk_id = 0
            self.__document_id = 0

    def ingest(self, text: str, max_seq_length: int = 512, metadata: dict = None) -> int:
        """
        Ingests the input text and returns the DOCUMENT ID (not chunk ID) associated with it in the database.
        """
        if not isinstance(text, str):
            raise TypeError("The 'text' argument must be a string.")
        if metadata and not isinstance(metadata, dict):
            raise TypeError("The 'metadata' argument must be a dict.")

        chunks = chop_and_chunk(text, max_seq_length=max_seq_length)
        encoded_chunks = self.__embed_model.embed(texts=chunks, device=self.device)  # this is a numpy array, where each row is the vector for each chunk in chunks
        
        keys = np.arange(self.__chunk_id, self.__chunk_id + len(chunks))
        self.__index.add(keys=keys, vectors=encoded_chunks)  # if you pass in ndarray, ndarray for keys, vectors, it processes in a batch add matching key to row (see USearch docs)

        for chunk in chunks:
            self.__texts[self.__chunk_id] = chunk
            self.__metadata[self.__chunk_id] = metadata or {}
            self.__metadata[self.__chunk_id]['document_id'] = self.__document_id
            self.__chunk_id += 1

        self.__document_id += 1
        self.save()

        return self.__document_id - 1

    def retrieve(self, text: str, top_k: int = 3, autocut: bool = False, autocut_amount: int = 25, get_metadata: bool = False, get_similarities: bool = False, progress: bool = False) -> dict:
        """
        Method to retrieve vectors given text. Will always return the text, and can specify what else
        we want to return with the get_THING flag. If we set autocut=True, top_k will function as the number of
        CLUSTERS to return, not results. autocut_amount is how many items we run the autocut algorithm over.
        """
        if not isinstance(text, str):
            raise TypeError("The 'text' argument must be a string.")
        if top_k <= 0:
            raise Exception("Please input k >= 1.")

        count = autocut_amount if autocut else top_k  # sets the amount of elements we want to autocut over here

        matches: Matches = self.__index.search(self.__embed_model.embed(texts=text, device=self.device), count=count, log=progress)
        matches = matches.to_list()

        indices = [match[0] for match in matches]  # indices of the top matches used to retrieve the text and metadata
        scores = [match[1] for match in matches]  # cosine similarity scores in order of descending similarity (ascending value when returned by usearch)

        if autocut:
            sim_diff = np.diff(scores)
            std_dev = np.std(sim_diff)
            cluster_idxs = np.where(sim_diff > std_dev)[0]  # indices marking the position of the END of each autocut cluster
            k = min(top_k, len(cluster_idxs))  # ensures that only pull a MAX of top_k clusters (can have less if no other relevant clusters found)
            if cluster_idxs.size > 0:
                endpoint = cluster_idxs[k - 1] + 1  # gets indices of elements in top k CLUSTERS
            else:
                endpoint = k
            indices = indices[0:endpoint]
            scores = scores[0:endpoint]
            texts: list = [self.__texts[idx] for idx in indices]

        else:
            texts: list = [self.__texts[idx] for idx in indices]

        results = {"texts": texts}
        if get_metadata:
            metadata: list = [self.__metadata[idx] for idx in indices]
            results["metadata"] = metadata
        if get_similarities:
            scores = [1 - score for score in scores]  # cosine similarity in order of descending similarity (descending value, 1 = perfect match)
            results["scores"] = scores
        return results

    def save(self):
        """
        Saves the database metadata and index files.
        """
        with open(self.__metadata_file, 'wb') as f:
            np.savez(f, texts=self.__texts, metadata=self.__metadata, chunk_id=self.__chunk_id, document_id=self.__document_id)
        self.__index.save(path_or_buffer=self.__index_file)

    def get_metadata_file(self) -> str:
        """
        Returns the file where metadata is located.
        """
        return self.__metadata_file

    def get_index_file(self) -> str:
        """
        Returns the file where the index is located.
        """
        return self.__index_file
    
    def get_metadata(self, keys: list[int] = []) -> dict:
        """
        Returns the metadata. Metadata is structured as ID -> metdata.
        """
        if keys:
            return {key: self.__metadata[key] for key in keys if key in self.__metadata}
        return self.__metadata

    def get_texts(self, keys: list[int] = []) -> dict:
        """
        Returns the texts. Text is structured as ID -> text.
        """
        if keys:
            return {key: self.self.__texts[key] for key in keys if key in self.__texts}
        return self.__texts
    
    def get_vectors(self, keys: list[int] = []) -> dict[int, np.ndarray]:
        """
        Returns all the raw vectors from the index file itself. If you specify a key, it will only return the vector from that key.
        If a key is not found, the corresponding vector will be an array of zeros. Return format is a dict of key -> vector.
        """
        target_keys = keys if keys else list(self.__texts.keys())
        return {key: self.__index.get(key) for key in target_keys}

    def clear(self) -> str:
        """
        Clears the database of all entries.
        """
        self.__index.clear()
        self.__texts = {}
        self.__metadata = {}
        self.__chunk_id = 0
        self.__document_id = 0

    def __len__(self) -> int:
        """
        Returns the total number of documents (and thus, vectors) in the database.
        """
        return len(self.__texts)
    
    def __str__(self) -> str:
        """
        Casts the vdb object to a string.
        """
        return f"VLite2(name={self.__name}, length={len(self)}, index_file={self.__index_file}, metadata_file={self.__metadata_file})"
 