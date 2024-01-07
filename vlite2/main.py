import numpy as np
from uuid import uuid4
from .model import EmbeddingModel
from .utils import chop_and_chunk, cos_sim
import datetime


class VLite:
    def __init__(self, collection_name=None, device='mps', model=None):
        if collection_name is None:
            current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            collection_name = f"vlite_{current_datetime}.npz"

        self.collection = collection_name
        self.device = device
        self.model = model if model else EmbeddingModel()
        try:
            with np.load(self.collection, allow_pickle=True) as data:
                self.texts = data['texts'].tolist()
                self.metadata = data['metadata'].tolist()
                self.vectors = data['vectors']
        except FileNotFoundError:
            self.texts = []
            self.metadata = {}
            self.vectors = np.empty((0, self.model.dimension))

    def add_vector(self, vector):
        self.vectors = np.vstack((self.vectors, vector))

    def get_similar_vectors(self, vector, top_k):
        sims = cos_sim(vector, self.vectors)
        sims = sims[0]
        top_k_idx = np.argsort(sims)[::-1][:top_k]
        return top_k_idx, sims[top_k_idx]

    def memorize(self, text, max_seq_length=512, id=None, metadata=None):
        id = id or str(uuid4())
        chunks = chop_and_chunk(text, max_seq_length=max_seq_length)
        encoded_data = self.model.embed(texts=chunks, device=self.device)
        self.vectors = np.vstack((self.vectors, encoded_data))
        for chunk in chunks:
            self.texts.append(chunk)
            idx = len(self.texts) - 1
            self.metadata[idx] = metadata or {}
            self.metadata[idx]['index'] = id or idx
        self.save()
        return id, self.vectors

    def remember(self, text=None, id=None, top_k=3, autocut=False, autocut_amount=50, return_metadata=False, return_similarities=False) -> tuple:
        """
        Method to remember vectors given text OR an ID. Will always return the text, and can specify what else
        we want to return with the return_THING flag. If we set autocut=True, top_k will function as the number of
        CLUSTERS to return, not results. autocut_amount is how many items we run the autocut algorithm over.
        """
        if not id and not text:
            raise Exception("Please input either text or ID to retrieve from.")
        if id:
            return self.metadata[id]
        if top_k <= 0:
            raise Exception("Please input k >= 1.")
        if text:
            sims = cos_sim(self.model.embed(texts=text, device=self.device), self.vectors)
            sims = sims[0]

            k = autocut_amount if autocut else top_k  # set default amount to run autocut over as 50 here
            k = min(k, len(sims))

            top_k_idx = np.argpartition(sims, -k)[-k:]  # pulls the UNSORTED indices for the top k similarities
            top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]  # sorts top k indices descending
            desc_similarities = sims[top_k_idx]

            if autocut:
                sim_diff = np.diff(desc_similarities) * -1
                std_dev = np.std(sim_diff)
                cluster_idxs = np.where(sim_diff > std_dev)[0]  # indices marking the end of each autocut cluster
                k = min(top_k, len(cluster_idxs))
                if cluster_idxs.size > 0:
                    top_k_idx = top_k_idx[0:cluster_idxs[k - 1] + 1]  # gets indices of elements in top k CLUSTERS
                else:
                    top_k_idx = top_k_idx[0:k]  # gets indices of elements in top k if there are no
                texts: list = [self.texts[idx] for idx in top_k_idx]

            else:
                texts: list = [self.texts[idx] for idx in top_k_idx]

            return_tuple = (texts,)
            if return_metadata:
                metadata: list = [self.metadata[idx] for idx in top_k_idx]
                return_tuple = return_tuple + (metadata,)
            if return_similarities:
                return_tuple = return_tuple + (desc_similarities,)
            return return_tuple

    def save(self):
        with open(self.collection, 'wb') as f:
            np.savez(f, texts=self.texts, metadata=self.metadata, vectors=self.vectors)
