# vlite

a blazing fast, lightweight, and simple vector database written in less than 200 lines of code.

## usage

```python
from vlite import VLite

db = VLite()

db.memorize(["hello world"]*5)

db.remember("adele")

```

## installation

```bash
pip install vlite
```

## about

![442f7062-ab69-4621-96ca-4ee66df06ffc](https://github.com/sdan/vlite/assets/22898443/fc36481c-f1f6-4973-8461-6aef3a04486d)

VLite is a vector database built for agents, ChatGPT Plugins, and other AI apps that need a fast and simple database to store vectors. 

It uses Apple's Metal Performance Shaders via Pytorch to accelerate vector loading and uses CPU threading to accelerate vector queries to reduce time spent copying vectors from the GPU(MPS) to the CPU.

I made this because I needed a vector db that was easy to use and be able to handle reading/writing millions of embeddings a day and the only good alternatives that ran locally crashed constantly and were slow.

### colors

taken from OpenAI's tiktoken repo, I added a visualize_tokens() function to visualize BPE tokens, i made visualize_tokens to handle the output of the tokenizer.encode() function since the currently supported embeddings are based on BERT and don't use the same tokenization as GPT-4.
