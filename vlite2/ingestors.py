import os
from .main import VLite2
import yaml
import re
import PyPDF2
import docx2txt

class Ingestor:
    def __init__(self, vlite2: VLite2) -> None:
        """
        Accept vlite2 instance as the constructor.
        """
        self.vlite2 = vlite2

    def __replaceNewlines(self, text: str) -> str:
        """
        Replace any sequence of 3 or more "\n" with just "\n\n" for splitting purposes.
        """
        return re.sub(r'\n{3,}', '\n\n', text)
    
    def processString(self, data: str, chunk_size: int = 128, source: str = 'string', verbose: bool = False):
        """
        Given some string data, ingest it.
        """
        snippets = self.__replaceNewlines(data).split("\n\n")

        # removing small irrelevant snippets of text by combining them all
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
        """
        Given a pdf file, ingest the data in it. Each page is a new entry, with the contents chunked up.
        """
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
        """
        Given a txt file, ingest the data in it. Each entry denoted by the text file split on two newlines. chunked to 128 characters by default.
        """
        if not filename.endswith('.txt'):
            raise ValueError("The file must be a txt")
        if not os.path.exists(filename):
            raise FileNotFoundError("The file does not exist.")
        
        with open(filename, "r") as f:
            data = f.read()

        self.processString(data, chunk_size, source=filename, verbose=verbose)

    def processDict(self, data: dict, chunk_size: int = 128, verbose: bool = False):
        """
        Given an input dictionary entry (NOT a list of dicts), process the entire thing into our vector database. Save as a yaml since it's more LLM-efficient.
        """

        if verbose:
            print(f"\n\n{'-' * 10}STARTED PROCESSING DICTIONARY OBJECT{'-' * 10}")
        data_yaml = str(yaml.dump(data))
        self.vlite2.ingest(text=data_yaml, metadata={"location": f"dictionary object with keys {list(data.keys())}", "content": data}, max_seq_length=chunk_size)
        if verbose:
            print(f"{'-' * 10}FINISHED PROCESSING DICTIONARY OBJECT{'-' * 10}\n\n")

    def processDocx(self, filename: str, chunk_size: int = 128, verbose: bool = False):
        """
        Given an input word document, process it into the vector database. Save all images in the word doc to their own directory.
        """
        if not filename.endswith('.docx'):
            raise ValueError("The file must be a .docx")

        if not os.path.exists(filename):
            raise FileNotFoundError("The file does not exist.")
        
        image_path = os.path.join(os.getcwd(), f"{filename}_images")
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        text = docx2txt.process(filename, image_path)

        self.processString(text, chunk_size, source=filename, verbose=verbose)
