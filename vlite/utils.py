import numpy as np
import itertools
from typing import List
from transformers import AutoTokenizer
import regex as re


def chop_and_chunk(text, max_seq_length):
    """
    Chop and chunk a text into smaller pieces of text. 
    
    Args:
    text: string, list of strings, or array of strings 
    max_seq_length: maximum length of the text
    """
    if isinstance(text, str):
        text = [text]
        
    chunks = []
    for t in text: 
        parts = re.split('\n+', t)  # split by newline
        
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


def visualize_tokens(token_values: List[str]) -> None:
        backgrounds = itertools.cycle(
            ["\u001b[48;5;{}m".format(i) for i in [167, 179, 185, 77, 80, 68, 134]]
        )
        interleaved = itertools.chain.from_iterable(zip(backgrounds, token_values))
        print(("".join(interleaved) + "\u001b[0m"))


def token_count(texts):
        tz = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', use_fast=True)
        tokens = 0
        for text in texts:
            tokens+=len(tz.tokenize(text, padding=True, truncation=True))
        return tokens
