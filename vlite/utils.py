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

class Retrieval:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', vlite2=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.vlite2 = vlite2

    def __replaceNewlines(self, text: str) -> str:
        return re.sub(r'\n{3,}', '\n\n', text)

    def chop_and_chunk(self, text, max_seq_length=256):
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

    def token_count(self, texts):
        tokens = 0
        for text in texts:
            tokens += len(self.tokenizer.tokenize(text, padding=True, truncation=True))
        return tokens

    def process_and_ingest(self, data: Union[str, dict], chunk_size: int = 128, source: str = 'string', verbose: bool = False):
        if isinstance(data, dict):
            self.processDict(data, chunk_size, verbose)
        elif isinstance(data, str):
            if data.endswith('.pdf'):
                self.processPDF(data, chunk_size, verbose)
            elif data.endswith('.txt'):
                self.processTXT(data, chunk_size, verbose)
            elif data.endswith('.docx'):
                self.processDocx(data, chunk_size, verbose)
            else:
                self.processString(data, chunk_size, source, verbose)
        else:
            raise ValueError("Unsupported data type. Please provide a string, dictionary, or a supported file path.")

    def processString(self, data: str, chunk_size: int = 128, source: str = 'string', verbose: bool = False):
        snippets = self.__replaceNewlines(data).split("\n\n")
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
        for i, info in enumerate(snippets):
            if verbose:
                print(f"\n{'-' * 10}PROCESSING SNIPPET {i + 1}{'-' * 10}\n")
            self.vlite2.ingest(text=info, metadata={"location": f"{source} snippet {i + 1}", "content": info}, max_seq_length=chunk_size)
        if verbose:
            print(f"\n\n{'-' * 10}FINISHED PROCESSING TEXT: {source}{'-' * 10}\n\n")
    
    def processPDF(self, filename: str, chunk_size: int = 128, verbose: bool = False):
        if not filename.endswith('.pdf'):
            raise ValueError("The file must be a pdf")
        if not os.path.exists(filename):
            raise FileNotFoundError("The file does not exist.")

        if verbose:
            print(f"\n\n{'-' * 10}STARTED PROCESSING PDF: {filename}{'-' * 10}\n\n")
        with open(filename, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num in range(len(pdf_reader.pages)):
                if verbose:
                    print(f"\n{'-' * 10}PROCESSING PAGE {page_num + 1}{'-' * 10}\n")
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                text = self.__replaceNewlines(text)
                self.vlite2.ingest(text=text, metadata={"location": f"{filename} page {page_num + 1}", "content": text}, max_seq_length=chunk_size)
        if verbose:
            print(f"\n\n{'-' * 10}FINISHED PROCESSING PDF: {filename}{'-' * 10}\n\n")

    def processTXT(self, filename: str, chunk_size: int = 128, verbose: bool = False):
        if not filename.endswith('.txt'):
            raise ValueError("The file must be a txt")
        if not os.path.exists(filename):
            raise FileNotFoundError("The file does not exist.")
        
        with open(filename, "r") as f:
            data = f.read()

        self.processString(data, chunk_size, source=filename, verbose=verbose)

    def processDict(self, data: dict, chunk_size: int = 128, verbose: bool = False):
        if verbose:
            print(f"\n\n{'-' * 10}STARTED PROCESSING DICTIONARY OBJECT{'-' * 10}")
        data_yaml = str(yaml.dump(data))
        self.vlite2.ingest(text=data_yaml, metadata={"location": f"dictionary object with keys {list(data.keys())}", "content": data}, max_seq_length=chunk_size)
        if verbose:
            print(f"{'-' * 10}FINISHED PROCESSING DICTIONARY OBJECT{'-' * 10}\n\n")

    def processDocx(self, filename: str, chunk_size: int = 128, verbose: bool = False):
        if not filename.endswith('.docx'):
            raise ValueError("The file must be a .docx")

        if not os.path.exists(filename):
            raise FileNotFoundError("The file does not exist.")
        
        image_path = os.path.join(os.getcwd(), f"{filename}_images")
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        text = docx2txt.process(filename, image_path)

        self.processString(text, chunk_size, source=filename, verbose=verbose)

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