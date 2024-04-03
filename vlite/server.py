from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional, Union
from vlite.main import VLite
from vlite.utils import process_file, process_pdf, process_webpage

app = FastAPI()
vlite = VLite(collection="vlite_server")

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

@app.post("/add", summary="Add text to the collection")
async def add_text(data: Union[TextData, List[TextData]]):
    """
    Add one or more texts to the VLite collection.

    - **data**: The text data to be added. It can be a single TextData object or a list of TextData objects.
        - **text**: The text content to be added.
        - **metadata** (optional): Additional metadata associated with the text.

    Returns:
    - A JSON object containing the message and the results of the add operation.
    """
    if isinstance(data, TextData):
        data = [data]

    texts = [item.text for item in data]
    metadatas = [item.metadata for item in data]
    results = vlite.add(texts, metadata=metadatas)
    return {"message": "Texts added successfully", "results": results}

@app.post("/add_file", summary="Add text from a file to the collection")
async def add_file(file: UploadFile = File(...)):
    """
    Add text from a file to the VLite collection.

    - **file**: The file to be uploaded and processed. Supported file types: .txt, .pdf, .docx, .csv, .pptx.

    Returns:
    - A JSON object containing the message and the results of the add operation.
    """
    file_path = await save_upload_file(file)
    chunks = process_file(file_path)
    results = vlite.add(chunks)
    return {"message": "File processed and added successfully", "results": results}

@app.post("/add_pdf", summary="Add text from a PDF file to the collection")
async def add_pdf(file: UploadFile = File(...), use_ocr: bool = False):
    """
    Add text from a PDF file to the VLite collection.

    - **file**: The PDF file to be uploaded and processed.
    - **use_ocr** (optional): Whether to use OCR for text extraction. Default is False.

    Returns:
    - A JSON object containing the message and the results of the add operation.
    """
    file_path = await save_upload_file(file)
    chunks = process_pdf(file_path, use_ocr=use_ocr)
    results = vlite.add(chunks)
    return {"message": "PDF processed and added successfully", "results": results}

@app.post("/add_webpage", summary="Add text from a webpage to the collection")
async def add_webpage(url: str):
    """
    Add text from a webpage to the VLite collection.

    - **url**: The URL of the webpage to be processed.

    Returns:
    - A JSON object containing the message and the results of the add operation.
    """
    chunks = process_webpage(url)
    results = vlite.add(chunks)
    return {"message": "Webpage processed and added successfully", "results": results}

@app.post("/retrieve", summary="Retrieve similar texts")
async def retrieve_text(request: RetrieveRequest):
    """
    Retrieve similar texts from the VLite collection based on the provided query text and metadata.

    - **request**: The retrieval request parameters.
        - **text** (optional): The query text for finding similar texts.
        - **top_k** (optional): The number of top similar texts to retrieve. Default is 5.
        - **metadata** (optional): Metadata to filter the retrieved texts.

    Returns:
    - A JSON object containing the results of the retrieval operation.
    """
    if request.text is None and request.metadata is None:
        raise HTTPException(status_code=400, detail="Either 'text' or 'metadata' must be provided")

    results = vlite.retrieve(text=request.text, top_k=request.top_k, metadata=request.metadata)
    return {"results": results}

@app.delete("/delete", summary="Delete items from the collection")
async def delete_texts(ids: Union[str, List[str]]):
    """
    Delete one or more items from the VLite collection based on their IDs.

    - **ids**: A single ID or a list of IDs of the items to delete.

    Returns:
    - A JSON object containing the message and the number of items deleted.
    """
    deleted_count = vlite.delete(ids)
    return {"message": f"{deleted_count} item(s) deleted successfully"}

@app.put("/update/{item_id}", summary="Update an item in the collection")
async def update_text(item_id: str, request: UpdateRequest):
    """
    Update an item in the VLite collection based on its ID.

    - **item_id**: The ID of the item to update.
    - **request**: The update request parameters.
        - **text** (optional): The updated text content of the item.
        - **metadata** (optional): The updated metadata of the item.
        - **vector** (optional): The updated embedding vector of the item.

    Returns:
    - A JSON object containing the message indicating the success of the update operation.
    """
    updated = vlite.update(item_id, text=request.text, metadata=request.metadata, vector=request.vector)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Item with ID '{item_id}' not found")
    return {"message": f"Item with ID '{item_id}' updated successfully"}

@app.get("/get", summary="Get items from the collection")
async def get_texts(ids: Optional[List[str]] = None, where: Optional[dict] = None):
    """
    Retrieve items from the VLite collection based on their IDs and/or metadata.

    - **ids** (optional): List of IDs to retrieve. If provided, only items with the specified IDs will be returned.
    - **where** (optional): Metadata filter to apply. Items matching the filter will be returned.

    Returns:
    - A JSON object containing the results of the retrieval operation.
    """
    results = vlite.get(ids=ids, where=where)
    return {"results": results}

@app.get("/count", summary="Get the count of items in the collection")
async def count_items():
    """
    Get the number of items in the VLite collection.

    Returns:
    - A JSON object containing the count of items in the collection.
    """
    count = vlite.count()
    return {"count": count}

@app.post("/save", summary="Save the collection to a file")
async def save_collection():
    """
    Save the current state of the VLite collection to a file.

    Returns:
    - A JSON object containing the message indicating the success of the save operation.
    """
    vlite.save()
    return {"message": "Collection saved successfully"}

@app.post("/clear", summary="Clear the collection")
async def clear_collection():
    """
    Clear the entire VLite collection, removing all items and resetting the attributes.

    Returns:
    - A JSON object containing the message indicating the success of the clear operation.
    """
    vlite.clear()
    return {"message": "Collection cleared successfully"}

@app.get("/info", summary="Get information about the collection")
async def get_info():
    """
    Get information about the VLite collection, including the number of items, collection file path, and the embedding model used.

    Returns:
    - A JSON object containing the collection information.
    """
    info = {
        "count": vlite.count(),
        "collection": vlite.collection,
        "model": str(vlite.model)
    }
    return info

@app.get("/dump", summary="Dump the collection data")
async def dump_data():
    """
    Dump the VLite collection data to a dictionary for serialization.

    Returns:
    - A JSON object containing the dumped collection data.
    """
    data = vlite.dump()
    return {"data": data}

async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Save the uploaded file to disk and return its file path.
    """
    file_path = f"uploads/{upload_file.filename}"
    with open(file_path, "wb") as file:
        file.write(await upload_file.read())
    return file_path