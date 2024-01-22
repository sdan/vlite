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
