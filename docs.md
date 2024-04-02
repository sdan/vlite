# vlite Documentation

vlite is a simple(and blazing fast) vector database. It allows you to store and retrieve data semantically using embeddings.

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
- `device` (optional): The device to use for embedding ('cpu' or 'cuda'). Default is 'cpu'.
- `model_name` (optional): The name of the embedding model to use. Default is 'mixedbread-ai/mxbai-embed-large-v1'.

### Data types supported

- `text`: A string containing the text data.
- `.txt`: A path to a text file locally.
- `.pdf/.docx`: A path to a PDF file locally.
- `.csv`: A path to a CSV file locally.
- `.pptx`: A path to a PPTX file locally.
- `webpage`: A URL to a webpage.


### Adding Text to the Collection

To add text to the collection, use the `add` method:

```python
vlite.add(data, metadata=None)
```

- `data`: The text data to be added. It can be a string, a dictionary containing text, id, and/or metadata, or a list of strings or dictionaries.
- `metadata` (optional): Additional metadata to be appended to each text entry.

The `add` method returns a list of tuples, each containing the ID of the added text, the updated vectors array, and the metadata.

### Retrieving Similar Texts

To retrieve similar texts from the collection, use the `retrieve` method:

```python
vlite.retrieve(text=None, top_k=5, metadata=None)
```

- `text` (optional): The query text for finding similar texts.
- `top_k` (optional): The number of top similar texts to retrieve. Default is 5.
- `metadata` (optional): Metadata to filter the retrieved texts.

The `retrieve` method returns a tuple containing a list of similar texts, their similarity scores, and metadata (if applicable).

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

The `get` method returns a list of retrieved items, each item being a tuple of (text, metadata).

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

## License

AGPL-3.0 License
