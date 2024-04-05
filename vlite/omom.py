import os
import struct
import json
from enum import Enum
from typing import List, Dict, Union
import numpy as np

class OmomSectionType(Enum):
    HEADER = 0
    EMBEDDINGS = 1
    CONTEXTS = 2
    METADATA = 3

class OmomFile:
    MAGIC_NUMBER = b"OMOM"
    VERSION = 1

    def __init__(self, file_path):
        self.file_path = file_path
        self.header = {
            "embedding_model": "default",
            "embedding_size": 0,
            "embedding_dtype": "float32",
            "context_length": 0,
        }
        self.embeddings = []
        self.contexts = []
        self.metadata = {}

    def set_header(self, embedding_model: str, embedding_size: int, embedding_dtype: str, context_length: int):
        self.header["embedding_model"] = embedding_model
        self.header["embedding_size"] = embedding_size
        self.header["embedding_dtype"] = embedding_dtype
        self.header["context_length"] = context_length

    def add_embedding(self, embedding: List[float]):
        self.embeddings.append(embedding)

    def add_context(self, context: str):
        self.contexts.append(context)

    def add_metadata(self, key: str, value: Union[int, float, str]):
        self.metadata[key] = value

    def save(self):
        with open(self.file_path, "wb") as file:
            file.write(self.MAGIC_NUMBER)
            file.write(struct.pack("<I", self.VERSION))

            header_json = json.dumps(self.header).encode("utf-8")
            file.write(struct.pack("<II", OmomSectionType.HEADER.value, len(header_json)))
            file.write(header_json)

            if self.embeddings:
                embeddings_data = b"".join(
                    struct.pack(f"<{len(emb)}f", *[float(x) if not np.isnan(x) else 0.0 for x in emb])
                    for emb in self.embeddings
                )
                file.write(struct.pack("<II", OmomSectionType.EMBEDDINGS.value, len(embeddings_data)))
                file.write(embeddings_data)

            contexts_data = b"".join(struct.pack("<I", len(context.encode("utf-8"))) + context.encode("utf-8") for context in self.contexts)
            file.write(struct.pack("<II", OmomSectionType.CONTEXTS.value, len(contexts_data)))
            file.write(contexts_data)

            metadata_json = json.dumps(self.metadata).encode("utf-8")
            file.write(struct.pack("<II", OmomSectionType.METADATA.value, len(metadata_json)))
            file.write(metadata_json)
        
    def load(self):
        try:
            with open(self.file_path, "rb") as file:
                # Read and verify header
                magic_number = file.read(len(self.MAGIC_NUMBER))
                if magic_number != self.MAGIC_NUMBER:
                    raise ValueError(f"Invalid magic number: {magic_number}")

                version = struct.unpack("<I", file.read(4))[0]
                if version != self.VERSION:
                    raise ValueError(f"Unsupported version: {version}")

                # Read sections
                while True:
                    section_header = file.read(8)
                    if not section_header:
                        break
                    section_type, section_length = struct.unpack("<II", section_header)

                    if section_type == OmomSectionType.HEADER.value:
                        header_json = file.read(section_length).decode("utf-8")
                        self.header = json.loads(header_json)
                    elif section_type == OmomSectionType.EMBEDDINGS.value:
                        embeddings_data = file.read(section_length)
                        if embeddings_data:
                            embedding_size = len(embeddings_data) // 4
                            self.embeddings = [
                                list(struct.unpack_from(f"<{embedding_size // len(self.embeddings)}f", embeddings_data, i * embedding_size))
                                for i in range(len(self.embeddings))
                            ] if self.embeddings else [list(struct.unpack_from(f"<{embedding_size}f", embeddings_data))]
                    elif section_type == OmomSectionType.CONTEXTS.value:
                        contexts_data = file.read(section_length)
                        self.contexts = []
                        offset = 0
                        while offset < len(contexts_data):
                            context_length = struct.unpack_from("<I", contexts_data, offset)[0]
                            offset += 4
                            try:
                                context = contexts_data[offset : offset + context_length].decode("utf-8")
                                self.contexts.append(context)
                            except UnicodeDecodeError as e:
                                print(f"Error decoding context: {e}")
                            offset += context_length
                    elif section_type == OmomSectionType.METADATA.value:
                        metadata_json = file.read(section_length).decode("utf-8")
                        self.metadata = json.loads(metadata_json)
                    else:
                        raise ValueError(f"Unknown section type: {section_type}")

        except FileNotFoundError:
            pass
    
    def __repr__(self):
        output = "OmomFile:\n\n"
        output += "Header:\n"
        for key, value in self.header.items():
            output += f"  {key}: {value}\n"
        output += "\nEmbeddings:\n"
        for i, embedding in enumerate(self.embeddings):
            output += f"  {i}: {embedding}\n"
        output += "\nContexts:\n"
        for i, context in enumerate(self.contexts):
            output += f"  {i}: {context}\n"
        output += "\nMetadata:\n"
        for key, value in self.metadata.items():
            output += f"  {key}: {value}\n"
        return output

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()
    
class Omom:
    def __init__(self, directory="omnoms"):
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get(self, user):
        return os.path.join(self.directory, f"{user}.omom")

    def create(self, user: str) -> OmomFile:
        file_path = self.get(user)
        return OmomFile(file_path)

    def read(self, user_id: str) -> OmomFile:
        file_path = self.get(user_id)
        return OmomFile(file_path)

    def delete(self, user_id: str):
        file_path = self.get(user_id)
        if os.path.exists(file_path):
            os.remove(file_path)