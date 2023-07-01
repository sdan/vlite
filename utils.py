import numpy as np
import torch
from torch import Tensor
import pickle
import pysbd
import PyPDF2

def chop_and_chunk(text, max_seq_length=256):
    if isinstance(text, str):
        if len(text) < max_seq_length*4:
            return [text]
        text = [text]
    segmenter = pysbd.Segmenter(language="en", clean=False)
    chunks = []
    for i, sentence in enumerate(text):
        if len(sentence) < max_seq_length*4:
            chunks.append(sentence)
        else:
            segs = segmenter.segment(sentence)
            buffer = "" # buffer for the current chunk
            for seg in segs:
                if len(buffer) + len(seg) > max_seq_length*4:
                    chunks.append(buffer)
                    buffer = ""
                buffer += seg
                if len(buffer) > max_seq_length*4:
                    chunks.append(buffer)
                    buffer = ""
                else:
                    buffer += ""
    return chunks

def cos_sim(a, b):
    # print("[+] A:", a.shape)
    # print("[+] B:", b.shape)
    b = b.T  # transpose b
    # print("[+] Transposed B:", b.shape)
    result = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b, axis=0))
    # print("[+] Result:", result.shape)
    return result


def load_file(pdf_path):
    extracted_text = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        for page_number in range(num_pages):
            page = reader.pages[page_number]
            extracted_text.append(page.extract_text())
    return extracted_text