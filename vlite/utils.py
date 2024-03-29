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



def __replaceNewlines(text: str) -> str:
    return re.sub(r'\n{3,}', '\n\n', text)

def chop_and_chunk(text, max_seq_length=256):
    if isinstance(text, str):
        text = [text]
    
    if all('\n' in t for t in text):
        return text
    
    chunks = []
    for t in text:
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

def token_count(texts, tokenizer):
    tokens = 0
    for text in texts:
        tokens += len(tokenizer.tokenize(text, padding=True, truncation=True))
    return tokens

def process_and_ingest(data: Union[str, dict], chunk_size: int = 128, source: str = 'string', verbose: bool = False):
    if isinstance(data, dict):
        return process_dict(data, chunk_size, verbose)
    elif isinstance(data, str):
        if data.endswith('.pdf'):
            return process_pdf(data, chunk_size, verbose)
        elif data.endswith('.txt'):
            return process_txt(data, chunk_size, verbose)
        elif data.endswith('.docx'):
            return process_docx(data, chunk_size, verbose)
        else:
            return process_string(data, chunk_size, source, verbose)
    else:
        raise ValueError("Unsupported data type. Please provide a string, dictionary, or a supported file path.")

def process_string(data: str, chunk_size: int = 128, source: str = 'string', verbose: bool = False):
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
            processed_pages.append({"text": text, "metadata": {"location": f"{filename} page {page_num + 1}", "content": text}})
    if verbose:
        print(f"\n\n{'-' * 10}FINISHED PROCESSING PDF: {filename}{'-' * 10}\n\n")
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
