from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional, Union
from vlite.main import VLite
from vlite.utils import process_file, process_pdf, process_webpage

app = FastAPI(
    title="VLite API",
    description="API for VLite, a simple vector database for text embedding and retrieval.",
    version="0.2.0",
)

vlite = VLite()

class TextData(BaseModel):
    text: str
    metadata: Optional[dict] = None

class RetrieveRequest(BaseModel):
    text: Optional[str] = None
    top_k: int = 5
    metadata: Optional[dict] = None

class UpdateRequest(BaseModel):
    text: Optional[str] = None
    metadata: Optional[dict] = None
    vector: Optional[List[float]] = None

@app.post("/add", response_model=List[tuple], summary="Add text to the collection")
async def add_text(data: Union[TextData, List[TextData]]):
    """
    Add text or a list of texts to the VLite collection.

    - **data**: The text data to be added. It can be a single TextData object or a list of TextData objects.
        - **text**: The text content.
        - **metadata** (optional): Additional metadata associated with the text.

    Returns:
    - A list of tuples containing the ID of the added text, the updated vectors array, and the metadata.
    """
    if isinstance(data, TextData):
        data = [data]
    texts = [item.text for item in data]
    metadatas = [item.metadata for item in data]
    results = vlite.add(texts, metadata=metadatas)
    return results

@app.post("/add_file", response_model=List[tuple], summary="Add text from a file to the collection")
async def add_file(file: UploadFile = File(...)):
    """
    Add text from a file to the VLite collection.

    - **file**: The file to be uploaded and processed.

    Returns:
    - A list of tuples containing the ID of the added text, the updated vectors array, and the metadata.
    """
    file_path = await save_upload_file(file)
    chunks = process_file(file_path)
    results = vlite.add(chunks)
    return results

@app.post("/add_pdf", response_model=List[tuple], summary="Add text from a PDF file to the collection")
async def add_pdf(file: UploadFile = File(...), use_ocr: bool = False):
    """
    Add text from a PDF file to the VLite collection.

    - **file**: The PDF file to be uploaded and processed.
    - **use_ocr** (optional): Whether to use OCR for text extraction. Default is False.

    Returns:
    - A list of tuples containing the ID of the added text, the updated vectors array, and the metadata.
    """
    file_path = await save_upload_file(file)
    chunks = process_pdf(file_path, use_ocr=use_ocr)
    results = vlite.add(chunks)
    return results

@app.post("/add_webpage", response_model=List[tuple], summary="Add text from a webpage to the collection")
async def add_webpage(url: str):
    """
    Add text from a webpage to the VLite collection.

    - **url**: The URL of the webpage to be processed.

    Returns:
    - A list of tuples containing the ID of the added text, the updated vectors array, and the metadata.
    """
    chunks = process_webpage(url)
    results = vlite.add(chunks)
    return results

@app.post("/retrieve", response_model=List[tuple], summary="Retrieve similar texts")
async def retrieve_text(request: RetrieveRequest):
    """
    Retrieve similar texts from the VLite collection based on the provided query text and metadata.

    - **request**: The retrieval request parameters.
        - **text** (optional): The query text for finding similar texts.
        - **top_k** (optional): The number of top similar texts to retrieve. Default is 5.
        - **metadata** (optional): Metadata to filter the retrieved texts.

    Returns:
    - A list of tuples containing the similar texts, their similarity scores, and metadata (if applicable).
    """
    if request.text is None and request.metadata is None:
        raise HTTPException(status_code=400, detail="Either 'text' or 'metadata' must be provided")

    results = vlite.retrieve(text=request.text, top_k=request.top_k, metadata=request.metadata)
    return results

@app.delete("/delete", response_model=int, summary="Delete items from the collection")
async def delete_texts(ids: Union[str, List[str]]):
    """
    Delete one or more items from the VLite collection based on their IDs.

    - **ids**: A single ID or a list of IDs of the items to delete.

    Returns:
    - The number of items deleted from the collection.
    """
    deleted_count = vlite.delete(ids)
    return deleted_count

@app.put("/update/{item_id}", response_model=bool, summary="Update an item in the collection")
async def update_text(item_id: str, request: UpdateRequest):
    """
    Update an item in the VLite collection based on its ID.

    - **item_id**: The ID of the item to update.
    - **request**: The update request parameters.
        - **text** (optional): The updated text content of the item.
        - **metadata** (optional): The updated metadata of the item.
        - **vector** (optional): The updated embedding vector of the item.

    Returns:
    - True if the item was successfully updated, False otherwise.
    """
    updated = vlite.update(item_id, text=request.text, metadata=request.metadata, vector=request.vector)
    return updated

@app.get("/get", response_model=List[tuple], summary="Get items from the collection")
async def get_texts(ids: Optional[List[str]] = None, where: Optional[dict] = None):
    """
    Retrieve items from the VLite collection based on their IDs and/or metadata.

    - **ids** (optional): List of IDs to retrieve. If provided, only items with the specified IDs will be returned.
    - **where** (optional): Metadata filter to apply. Items matching the filter will be returned.

    Returns:
    - A list of tuples containing the retrieved items, each item being a tuple of (text, metadata).
    """
    results = vlite.get(ids=ids, where=where)
    return results

@app.get("/count", response_model=int, summary="Get the count of items in the collection")
async def count_items():
    """
    Get the number of items in the VLite collection.

    Returns:
    - The count of items in the collection.
    """
    count = vlite.count()
    return count

@app.post("/save", response_model=None, summary="Save the collection to a file")
async def save_collection():
    """
    Save the current state of the VLite collection to a file.
    """
    vlite.save()

@app.post("/clear", response_model=None, summary="Clear the collection")
async def clear_collection():
    """
    Clear the entire VLite collection, removing all items and resetting the attributes.
    """
    vlite.clear()

@app.get("/info", response_model=dict, summary="Get information about the collection")
async def get_info():
    """
    Get information about the VLite collection, including the number of items, collection file path, and the embedding model used.

    Returns:
    - A dictionary containing the collection information.
    """
    info = {
        "count": vlite.count(),
        "collection": vlite.collection,
        "model": str(vlite.model)
    }
    return info

@app.get("/dump", response_model=dict, summary="Dump the collection data")
async def dump_data():
    """
    Dump the VLite collection data to a dictionary for serialization.

    Returns:
    - A dictionary containing the dumped collection data.
    """
    data = vlite.dump()
    return data

async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Save the uploaded file to disk and return its file path.
    """
    file_path = f"uploads/{upload_file.filename}"
    with open(file_path, "wb") as file:
        file.write(await upload_file.read())
    return file_path