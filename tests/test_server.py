from fastapi.testclient import TestClient
from server import app, vlite

client = TestClient(app)

def test_add_text():
    response = client.post("/add", json={"text": "This is a test text.", "metadata": {"source": "test"}})
    assert response.status_code == 200
    assert response.json() == {"message": "Texts added successfully", "results": [[None, None, {"source": "test"}]]}

def test_retrieve_text():
    client.post("/add", json={"text": "This is a test text.", "metadata": {"source": "test"}})
    response = client.post("/retrieve", json={"text": "test"})
    assert response.status_code == 200
    assert len(response.json()["results"]) > 0

def test_delete_texts():
    client.post("/add", json={"text": "This is a test text.", "metadata": {"id": "test_id"}})
    response = client.delete("/delete", json=["test_id"])
    assert response.status_code == 200
    assert response.json() == {"message": "1 item(s) deleted successfully"}

def test_update_text():
    client.post("/add", json={"text": "This is a test text.", "metadata": {"id": "test_id"}})
    response = client.put("/update/test_id", json={"text": "Updated text"})
    assert response.status_code == 200
    assert response.json() == {"message": "Item with ID 'test_id' updated successfully"}

def test_get_texts():
    client.post("/add", json={"text": "This is a test text.", "metadata": {"id": "test_id", "source": "test"}})
    response = client.get("/get", params={"where": {"source": "test"}})
    assert response.status_code == 200
    assert len(response.json()["results"]) > 0

def test_count_items():
    response = client.get("/count")
    assert response.status_code == 200
    assert response.json()["count"] >= 0

def test_clear_collection():
    response = client.post("/clear")
    assert response.status_code == 200
    assert response.json() == {"message": "Collection cleared successfully"}