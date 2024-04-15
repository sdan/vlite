# vlite Documentation
vlite is a simple and blazing fast vector database. It allows you to store and retrieve data semantically using embeddings.

## Installation
```bash
pip install vlite
```

### Installation with PDF OCR Support
To enable PDF OCR support, you need to install the `vlite[ocr]` extra:
```bash
pip install vlite[ocr]
```

## Usage
### Importing VLite
```python
from vlite import VLite
```

### Creating a VLite Instance
```python
vlite = VLite(collection="my_collection")
```
- `collection` (optional): The name of the collection file. If not provided, a default name will be generated based on the current timestamp.
- `device` (optional): The device to use for embedding ('cpu', 'mps', or 'cuda'). Default is 'cpu'. 'mps' uses PyTorch's Metal Performance Shaders on M1 macs, 'cuda' uses a NVIDIA GPU for embedding generation.
- `model_name` (optional): The name of the embedding model to use. Default is 'mixedbread-ai/mxbai-embed-large-v1'.

### Data Types Supported
- `text`: A string containing the text data.
- `.txt`: A path to a text file locally.
- `.pdf/.docx`: A path to a PDF file locally.
- `.csv`: A path to a CSV file locally.
- `webpage`: A URL to a webpage.

### Adding Text to the Collection
To add text to the collection, use the `add` method:
```python
vlite.add(data, metadata=None, item_id=None, need_chunks=False, fast=True)
```
- `data`: The text data to be added. It can be a string, a dictionary containing text, id, and/or metadata, or a list of strings or dictionaries.
- `metadata` (optional): Additional metadata to be appended to each text entry.
- `item_id` (optional): A unique identifier for the text item being added. If not provided, a random UUID will be generated.
- `need_chunks` (optional): Whether to split the text into chunks. Default is `False`.
- `fast` (optional): Whether to use a faster chunking method. Default is `True`.

The `add` method returns a list of tuples, each containing the ID of the added text, the binary encoded embedding, and the metadata.

### Retrieving Similar Texts
To retrieve similar texts from the collection, use the `retrieve` method:
```python
vlite.retrieve(text=None, top_k=5, metadata=None, return_scores=False)
```
- `text`: The query text for finding similar texts.
- `top_k` (optional): The number of top similar texts to retrieve. Default is 5.
- `metadata` (optional): Metadata to filter the retrieved texts.
- `return_scores` (optional): Whether to return the similarity scores along with the retrieved texts. Default is `False`.

The `retrieve` method returns a list of tuples, each containing the index, text, metadata, and optionally the similarity score (if `return_scores` is `True`) of the retrieved texts.

### Deleting Items
To delete items from the collection, use the `delete` method:
```python
vlite.delete(ids)
```
- `ids`: A single ID or a list of IDs of the items to delete.

The `delete` method returns the number of items deleted from the collection.

### Updating Items
To update an item in the collection, use the `update` method:
```python
vlite.update(id, text=None, metadata=None, vector=None)
```
- `id`: The ID of the item to update.
- `text` (optional): The updated text content of the item.
- `metadata` (optional): The updated metadata of the item.
- `vector` (optional): The updated embedding vector of the item.

The `update` method returns `True` if the item was successfully updated, `False` otherwise.

### Retrieving Items
To retrieve items from the collection based on IDs and/or metadata, use the `get` method:
```python
vlite.get(ids=None, where=None)
```
- `ids` (optional): List of IDs to retrieve. If provided, only items with the specified IDs will be returned.
- `where` (optional): Metadata filter to apply. Items matching the filter will be returned.

The `get` method returns a list of retrieved items, each item being a tuple of (id, text, metadata).

### Setting Item Attributes
To set attributes for an item in the collection, use the `set` method:
```python
vlite.set(id, text=None, metadata=None, vector=None)
```
- `id`: The ID of the item to set attributes for.
- `text` (optional): The text content of the item.
- `metadata` (optional): The metadata of the item.
- `vector` (optional): The embedding vector of the item.

If the item with the specified ID exists, it will be updated with the provided attributes. If the item does not exist, a new item will be created with the provided attributes.

### Counting Items
To get the number of items in the collection, use the `count` method:
```python
vlite.count()
```
The `count` method returns the count of items in the collection.

### Saving the Collection
To save the current state of the collection to a file, use the `save` method:
```python
vlite.save()
```
The `save` method saves the collection to the specified file.

### Clearing the Collection
To clear the entire collection, removing all items and resetting the attributes, use the `clear` method:
```python
vlite.clear()
```
The `clear` method clears the collection and saves the changes.

### Getting Collection Information
To print information about the collection, including the number of items, collection file path, and the embedding model used, use the `info` method:
```python
vlite.info()
```
The `info` method prints the collection information.

### Dumping Collection Data
To dump the collection data to a dictionary for serialization, use the `dump` method:
```python
vlite.dump()
```
The `dump` method returns a dictionary containing the collection data.

## CTX File Format
vlite uses the CTX (Context) file format for efficient storage and retrieval of embeddings and associated data. The CTX file format consists of the following sections:

1. **Header**: Contains metadata about the embedding model, embedding size, data type, and context length.
2. **Embeddings**: Stores the binary embeddings as a contiguous block of memory.
3. **Contexts**: Stores the associated text contexts for each embedding.
4. **Metadata**: Stores additional metadata associated with each embedding.

The CTX file format is designed to be memory-efficient and allows for fast loading and saving of embeddings and associated data.

### Creating a CTX File
To create a new CTX file, use the `create` method of the `Ctx` class:
```python
from vlite.ctx import Ctx

ctx = Ctx()
with ctx.create("example") as ctx_file:
    # Set header information
    ctx_file.set_header(
        embedding_model="example_model",
        embedding_size=64,
        embedding_dtype="binary",
        context_length=100
    )
    
    # Add embeddings, contexts, and metadata
    ctx_file.add_embedding([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, ...])
    ctx_file.add_context("This is an example context.")
    ctx_file.add_metadata("key", "value")
```

### Reading a CTX File
To read an existing CTX file, use the `read` method of the `Ctx` class:
```python
from vlite.ctx import Ctx

ctx = Ctx()
with ctx.read("example") as ctx_file:
    # Access header information
    header = ctx_file.header
    
    # Access embeddings, contexts, and metadata
    embeddings = ctx_file.embeddings
    contexts = ctx_file.contexts
    metadata = ctx_file.metadata
```

### Deleting a CTX File
To delete a CTX file, use the `delete` method of the `Ctx` class:
```python
from vlite.ctx import Ctx

ctx = Ctx()
ctx.delete("example")
```

## License
AGPL-3.0 License