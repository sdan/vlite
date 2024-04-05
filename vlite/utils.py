import os
import re
import PyPDF2
import docx2txt
import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import List
import tiktoken

try:
    from surya.ocr import run_ocr
    from surya.model.detection import segformer
    from surya.model.recognition.model import load_model
    from surya.model.recognition.processor import load_processor
    from surya.input.load import load_from_file, load_pdf

except ImportError:
    run_ocr = None

def chop_and_chunk(text, max_seq_length=512, fast=False):
    """
    Chop text into chunks of max_seq_length tokens or max_seq_length*4 characters (fast mode).
    """
    if isinstance(text, str):
        text = [text]
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []
    print(f"Length of text: {len(text)}")
    print(f"Original text: {text}")
    for t in text:
        if fast:
            chunk_size = max_seq_length * 4
            chunks.extend([t[i:i + chunk_size] for i in range(0, len(t), chunk_size)])
        else:
            token_ids = enc.encode(t, disallowed_special=())
            num_tokens = len(token_ids)
            if num_tokens <= max_seq_length:
                chunks.append(t)
            else:
                for i in range(0, num_tokens, max_seq_length):
                    chunk = enc.decode(token_ids[i:i + max_seq_length])
                    chunks.append(chunk)
    print("Chopped text into these chunks:", chunks)
    print(f"Chopped text into {len(chunks)} chunks.")
    return chunks

def process_pdf(file_path: str, chunk_size: int = 512, use_ocr: bool = False, langs: List[str] = None) -> List[str]:
    """
    Process a PDF file and return a list of text chunks.

    Args:
        file_path (str): The path to the PDF file.
        chunk_size (int, optional): The maximum number of tokens in each chunk. Defaults to 512.
        use_ocr (bool, optional): Whether to use OCR for text extraction. Defaults to False.
        langs (List[str], optional): The languages to use for OCR. Defaults to ['en'] if not provided.

    Returns:
        List[str]: A list of text chunks.
    """
    if use_ocr:
        if run_ocr is None:
            raise ImportError("OCR functionality is not available. Please install vlite with OCR support: pip install vlite[ocr]")
        
        if langs is None:
            langs = ['en']  # Default language if not provided
            
        print(f"Using OCR with languages: {langs}")
        
        det_processor, det_model = segformer.load_processor(), segformer.load_model()
        rec_model, rec_processor = load_model(), load_processor()
        image, _ = load_pdf(file_path, max_pages=len(file_path), start_page=0)        
        langs = ["en"] * len(image) 
        predictions = run_ocr(image, langs, det_model, det_processor, rec_model, rec_processor)
        print(predictions)
        text = [' '.join(result.text for result in prediction.text_lines) for prediction in predictions]
    else:
        print(f"Not using OCR for {file_path}")
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
    
    return chop_and_chunk(text, chunk_size)

def process_txt(file_path: str, chunk_size: int = 512) -> List[str]:
    """
    Process a text file and return a list of text chunks.

    Args:
        file_path (str): The path to the text file.
        chunk_size (int, optional): The maximum number of tokens in each chunk. Defaults to 512.

    Returns:
        List[str]: A list of text chunks.
    """
    with open(file_path, 'r') as file:
        text = file.read()
    
    return chop_and_chunk(text, chunk_size)

def process_docx(file_path: str, chunk_size: int = 512) -> List[str]:
    """
    Process a Word document (.docx) and return a list of text chunks.

    Args:
        file_path (str): The path to the Word document.
        chunk_size (int, optional): The maximum number of tokens in each chunk. Defaults to 512.

    Returns:
        List[str]: A list of text chunks.
    """
    text = docx2txt.process(file_path)
    return chop_and_chunk(text, chunk_size)

def process_csv(file_path: str) -> List[str]:
    """
    Process a CSV file and return a list of rows as strings.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        List[str]: A list of rows as strings.
    """
    df = pd.read_csv(file_path)
    rows = df.astype(str).values.tolist()
    return rows

# def process_pptx(file_path: str, chunk_size: int = 512) -> List[str]:
#     """
#     Process a PowerPoint presentation (.pptx) and return a list of text chunks.

#     Args:
#         file_path (str): The path to the PowerPoint presentation.
#         chunk_size (int, optional): The maximum number of tokens in each chunk. Defaults to 512.

#     Returns:
#         List[str]: A list of text chunks.
#     """
#     presentation = Presentation(file_path)
#     text = ""
#     for slide in presentation.slides:
#         for shape in slide.shapes:
#             if hasattr(shape, 'text'):
#                 text += shape.text + "\n"
    
#     return chop_and_chunk(text, chunk_size)

def process_webpage(url: str, chunk_size: int = 512) -> List[str]:
    """
    Process a webpage and return a list of text chunks.

    Args:
        url (str): The URL of the webpage.
        chunk_size (int, optional): The maximum number of tokens in each chunk. Defaults to 512.

    Returns:
        List[str]: A list of text chunks.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    return chop_and_chunk(text, chunk_size)

def process_file(file_path: str, chunk_size: int = 512) -> List[str]:
    """
    Process a file based on its extension and return a list of text chunks.

    Args:
        file_path (str): The path to the file.
        chunk_size (int, optional): The maximum number of tokens in each chunk. Defaults to 512.

    Returns:
        List[str]: A list of text chunks.

    Raises:
        ValueError: If the file type is not supported.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    if extension == '.pdf':
        return process_pdf(file_path, chunk_size)
    elif extension == '.txt':
        return process_txt(file_path, chunk_size)
    elif extension == '.docx':
        return process_docx(file_path, chunk_size)
    elif extension == '.csv':
        return process_csv(file_path, chunk_size)
    elif extension == '.pptx':
        return process_pptx(file_path, chunk_size)
    else:
        raise ValueError(f"Unsupported file type: {extension}")
    
    
## Other functions

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

def count_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    token_ids = enc.encode(text, disallowed_special=())
    return len(token_ids)