from fastapi.testclient import TestClient
from vlite.server import app, vlite
import os
import numpy as np
from vlite.utils import process_pdf

client = TestClient(app)

def test_add_text():
    data = {"text": "This is a test text.", "metadata": {"source": "test"}}
    response = client.post("/add", json=[data])
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert isinstance(response.json()[0][0], str)
    assert isinstance(response.json()[0][1], list)
    assert response.json()[0][2] == {"source": "test"}

def test_add_multiple_texts():
    texts = [
        "This is the first text.",
        "This is the second text.",
        "This is the third text."
    ]
    metadata = {"source": "example", "tags": ["text", "example"]}
    response = client.post("/add", json=[{"text": text, "metadata": metadata} for text in texts])
    assert response.status_code == 200
    assert len(response.json()) == 3

def test_add_text_with_custom_id():
    text = "This is a text with custom ID."
    metadata = {"id": "custom_id", "source": "example", "tags": ["text", "example"]}
    response = client.post("/add", json=[{"text": text, "metadata": metadata}])
    assert response.status_code == 200
    assert response.json()[0][2]["id"] == "custom_id"

def test_retrieve_similar_texts():
    vlite.clear()
    texts = [
        "This is the first text.",
        "This is the second text.",
        "This is the third text."
    ]
    metadata = {"source": "example", "tags": ["text", "example"]}
    client.post("/add", json=[{"text": text, "metadata": metadata} for text in texts])
    
    query = "What is the text about?"
    response = client.post("/retrieve", json={"text": query, "top_k": 2})
    assert response.status_code == 200
    assert len(response.json()) == 2

def test_retrieve_text_by_id():
    vlite.clear()
    text = "This is a text with custom ID."
    metadata = {"id": "custom_id", "source": "example"}
    client.post("/add", json=[{"text": text, "metadata": metadata}])
    
    response = client.get("/get", params={"ids": ["custom_id"]})
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0][0] == text

def test_update_text():
    vlite.clear()
    text = "This is a text to be updated."
    metadata = {"id": "update_id", "source": "example"}
    client.post("/add", json=[{"text": text, "metadata": metadata}])
    
    updated_text = "This is the updated text."
    updated_metadata = {"source": "updated"}
    response = client.put("/update/update_id", json={"text": updated_text, "metadata": updated_metadata})
    assert response.status_code == 200
    assert response.json() == True

def test_delete_text():
    vlite.clear()
    text = "This is a text to be deleted."
    metadata = {"id": "delete_id", "source": "example"}
    client.post("/add", json=[{"text": text, "metadata": metadata}])
    
    response = client.delete("/delete", json=["delete_id"])
    assert response.status_code == 200
    assert response.json() == 1

def test_get_texts_by_ids():
    vlite.clear()
    texts = [
        "This is the first text.",
        "This is the second text.",
        "This is the third text."
    ]
    metadata = {"source": "example", "tags": ["text", "example"]}
    client.post("/add", json=[{"text": text, "metadata": metadata} for text in texts])
    
    response = client.get("/get", params={"ids": ["0", "1"]})
    assert response.status_code == 200
    assert len(response.json()) == 2

def test_get_texts_by_metadata():
    vlite.clear()
    texts = [
        "This is the first text.",
        "This is the second text.",
        "This is the third text."
    ]
    metadata = {"source": "example", "tags": ["text", "example"]}
    client.post("/add", json=[{"text": text, "metadata": metadata} for text in texts])
    
    response = client.get("/get", params={"where": {"source": "example"}})
    assert response.status_code == 200
    assert len(response.json()) == 3

def test_set_metadata():
    vlite.clear()
    text = "This is a text to set metadata."
    client.post("/add", json=[{"text": text}])
    
    response = client.put("/update/0", json={"metadata": {"updated": True}})
    assert response.status_code == 200
    assert response.json() == True

def test_set_text():
    vlite.clear()
    text = "This is a text to be updated."
    client.post("/add", json=[{"text": text}])
    
    updated_text = "This is the updated text."
    response = client.put("/update/0", json={"text": updated_text})
    assert response.status_code == 200
    assert response.json() == True

def test_set_vector():
    vlite.clear()
    text = "This is a text to set vector."
    client.post("/add", json=[{"text": text}])
    
    new_vector = np.random.rand(vlite.model.dimension).tolist()
    response = client.put("/update/0", json={"vector": new_vector})
    assert response.status_code == 200
    assert response.json() == True

def test_count_items():
    vlite.clear()
    texts = [
        "This is the first text.",
        "This is the second text.",
        "This is the third text."
    ]
    client.post("/add", json=[{"text": text} for text in texts])
    
    response = client.get("/count")
    assert response.status_code == 200
    assert response.json() == 3

def test_clear_collection():
    vlite.clear()
    texts = [
        "This is the first text.",
        "This is the second text.",
        "This is the third text."
    ]
    client.post("/add", json=[{"text": text} for text in texts])
    
    response = client.post("/clear")
    assert response.status_code == 200
    
    count_response = client.get("/count")
    assert count_response.status_code == 200
    assert count_response.json() == 0

def test_process_pdf():
    vlite.clear()
    pdf_path = "data/attention.pdf"
    response = client.post("/add_pdf", files={"file": open(pdf_path, "rb")})
    assert response.status_code == 200
    assert len(response.json()) > 0

def test_query_pdf():
    vlite.clear()
    pdf_path = "data/attention.pdf"
    client.post("/add_pdf", files={"file": open(pdf_path, "rb")})
    
    query = "What is attention?"
    response = client.post("/retrieve", json={"text": query, "top_k": 3})
    assert response.status_code == 200
    assert len(response.json()) == 3