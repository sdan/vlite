import os
import yaml
import re
import PyPDF2
import docx2txt
import numpy as np
import pysbd
import itertools
from typing import List, Union
from transformers import AutoTokenizer, AutoModel
import tiktoken
import uuid
def chop_and_chunk(text, max_seq_length=512):
    if isinstance(text, str):
        text = [text]

    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []

    for t in text:
        token_ids = enc.encode(t, disallowed_special=())
        start_idx = 0
        while start_idx < len(token_ids):
            end_idx = min(start_idx + max_seq_length, len(token_ids))
            chunk = enc.decode(token_ids[start_idx:end_idx])
            chunks.append(chunk)
            start_idx = end_idx

    return chunks

def replace_newlines(text: str) -> str:
        """
        Replace any sequence of 3 or more "\n" with just "\n\n" for splitting purposes.
        """
        return re.sub(r'\n{3,}', '\n\n', text)

def process_string(data: str, chunk_size: int = 512, source: str = 'string', verbose: bool = False):
    snippets = replace_newlines(data).split("\n\n")
    merged_snippets = []
    previous_snippet = ""
    for snippet in snippets:
        if previous_snippet and len(snippet) < chunk_size:
            merged_snippets[-1] += " " + snippet
        else:
            merged_snippets.append(snippet)
            previous_snippet = snippet
    snippets = merged_snippets
    
    if verbose:
        print(f"\n\n{'-' * 10}STARTED PROCESSING TEXT FROM: {source}{'-' * 10}\n\n")
    processed_snippets = []
    for i, info in enumerate(snippets):
        if verbose:
            print(f"\n{'-' * 10}PROCESSING SNIPPET {i + 1}{'-' * 10}\n")
        processed_snippets.append({"text": info, "metadata": {"location": f"{source} snippet {i + 1}", "content": info}})
    if verbose:
        print(f"\n\n{'-' * 10}FINISHED PROCESSING TEXT: {source}{'-' * 10}\n\n")
    return processed_snippets

def process_pdf(filename: str, chunk_size: int = 128, verbose: bool = False):
    if not filename.endswith('.pdf'):
        raise ValueError("The file must be a pdf")
    if not os.path.exists(filename):
        raise FileNotFoundError("The file does not exist.")

    if verbose:
        print(f"\n\n{'-' * 10}STARTED PROCESSING PDF: {filename}{'-' * 10}\n\n")
    with open(filename, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        processed_pages = []
        for page_num in range(len(pdf_reader.pages)):
            if verbose:
                print(f"\n{'-' * 10}PROCESSING PAGE {page_num + 1}{'-' * 10}\n")
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            text = replace_newlines(text)
            processed_pages.append({"id": str(uuid.uuid4()), "text": text, "metadata": {"location": f"{filename} page {page_num + 1}", "content": text}})
    if verbose:
        print(f"\n\n{'-' * 10}FINISHED PROCESSING PDF: {filename}{'-' * 10}\n\n")
    
    print(f"Type of result: {type(processed_pages)}")
    # some samples of the processed pages
    print(f"Sample of processed pages: {processed_pages[:2]}")
    
    return processed_pages

def process_txt(filename: str, chunk_size: int = 128, verbose: bool = False):
    if not filename.endswith('.txt'):
        raise ValueError("The file must be a txt")
    if not os.path.exists(filename):
        raise FileNotFoundError("The file does not exist.")
    
    with open(filename, "r") as f:
        data = f.read()

    return process_string(data, chunk_size, source=filename, verbose=verbose)

def cos_sim(a, b):
    sims = a @ b.T
    sims /= np.linalg.norm(a) * np.linalg.norm(b, axis=1)
    return sims

def visualize_tokens(token_values: List[str]) -> None:
    backgrounds = itertools.cycle(
        ["\u001b[48;5;{}m".format(i) for i in [167, 179, 185, 77, 80, 68, 134]]
    )
    interleaved = itertools.chain.from_iterable(zip(backgrounds, token_values))
    print(("".join(interleaved) + "\u001b[0m"))

def load_file(pdf_path):
    extracted_text = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in iter(reader.pages):
            extracted_text.append(page.extract_text())  
    return extracted_text
