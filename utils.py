import numpy as np
import torch
from torch import Tensor
import pickle
import pysbd
import PyPDF2

# def chop_and_chunk(text, max_seq_length=256):
#     if isinstance(text, str):
#         if len(text) < max_seq_length*4:
#             return [text]
#         text = [text]
#     segmenter = pysbd.Segmenter(language="en", clean=False)
#     chunks = []
#     for i, sentence in enumerate(text):
#         if len(sentence) < max_seq_length*4:
#             chunks.append(sentence)
#         else:
#             segs = segmenter.segment(sentence)
#             buffer = "" # buffer for the current chunk
#             for seg in segs:
#                 if len(buffer) + len(seg) > max_seq_length*4:
#                     chunks.append(buffer)
#                     buffer = ""
#                 buffer += seg
#                 if len(buffer) > max_seq_length*4:
#                     chunks.append(buffer)
#                     buffer = ""
#                 else:
#                     buffer += ""
#     return chunks

def chop_and_chunk(text, max_seq_length=256):
    """
    Chop and chunk a text into smaller pieces of text. 
    
    Args:
    text: string, list of strings, or array of strings 
    max_seq_length: maximum length of the text
    """
    if isinstance(text, str):
        text = [text]
        
    # If text is already split into chunks by newlines, simply return it 
    if all('\n' in t for t in text):
        return text 
        
    chunks = []
    for t in text: 
        # Split by newlines 
        parts = re.split('\n+', t)  
        
        for p in parts:
            tokens = p.split()
            chunk = ''
            count = 0
            for t in tokens:
                if count + len(t) < max_seq_length:
                    count += len(t) 
                    chunk += t + ' '
                else:
                    chunks.append(chunk.strip())
                    count = 0
                    chunk = ''
            if chunk != '':
                chunks.append(chunk.strip())
    return chunks
    
def cos_sim(a, b):
    sims = a @ b.T
    sims /= np.linalg.norm(a) * np.linalg.norm(b, axis=1) 
    return sims


def load_file(pdf_path):
    extracted_text = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in iter(reader.pages):
            extracted_text.append(page.extract_text())  
    return extracted_text
