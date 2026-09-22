import numpy as np
import pytest

from vlite import VLite
from vlite.main import BYTES, hamming_topk


@pytest.fixture(scope="module")
def model_db(tmp_path_factory):
    return VLite("unit", directory=tmp_path_factory.mktemp("contexts"))  # loads the model once for the whole file


@pytest.fixture
def db(model_db):
    model_db.clear()
    return model_db


def test_hamming_is_popcount():
    # one flipped high bit must beat two flipped low bits (a sum of XOR'd bytes gets this backwards: 128 vs 3)
    q = np.zeros(BYTES, np.uint8)
    bits = np.zeros((2, BYTES), np.uint8)
    bits[0, 0], bits[1, 0] = 0b1000_0000, 0b0000_0011
    top, dist = hamming_topk(bits, q, 2)
    assert top.tolist() == [0, 1] and dist.tolist() == [1, 2]
    assert hamming_topk(bits[:0], q, 5)[0].size == 0  # empty collection


def test_retrieve_is_semantic(db):
    db.add(["The cat curled up on the windowsill and purred.",
            "Stocks fell sharply after the central bank raised rates.",
            "Boil the pasta for nine minutes, then toss it with garlic and oil."])
    assert "cat" in db.retrieve("a sleepy kitten", top_k=1)[0][1]


def test_long_text_is_chunked_not_truncated(db):
    text = " ".join(f"word{i}" for i in range(3000))
    [item] = db.add(text)
    rows = db.get(ids=item)
    assert len(rows) > 1 and db.count() == len(rows)
    assert rows[-1][1].endswith("word2999")  # the tail survives
    assert all(len(db.tokenizer(t)["input_ids"]) <= 512 for _, t, _ in rows)


def test_where_filters_before_search(db):
    db.add(["cats are great"] * 5 + ["dogs are great"], metadata=[{"pet": "cat"}] * 5 + [{"pet": "dog"}])
    results = db.retrieve("cats", top_k=3, where={"pet": "dog"})
    assert [r[2] for r in results] == [{"pet": "dog"}]


def test_save_delete_load_roundtrip(db):
    a = db.add(["alpha", "beta", "gamma"])
    db.save()
    b = db.add(["delta", "epsilon"])
    db.save()
    assert db.delete(a[0]) == 1
    db.save()
    snapshot = (list(db.ids), list(db.texts), db.bits.copy())
    db.ids, db.texts, db.metadata, db.bits = [], [], [], db.bits[:0]
    db.load()
    assert db.ids == snapshot[0] == a[1:] + b  # deleted row stays deleted, nothing duplicated
    assert db.texts == snapshot[1] == ["beta", "gamma", "delta", "epsilon"]
    assert np.array_equal(db.bits, snapshot[2])
