from setuptools import setup, find_packages

setup(
    name='vlite',
    version='0.2.0',
    author='Surya Dantuluri',
    author_email='surya@suryad.com',
    description='A simple vector database that stores vectors in a numpy array.',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'PyPDF2',
        'docx2txt',
        'pandas',
        'Requests',
        'beautifulsoup4',
        'llama-cpp-python',
        'huggingface_hub',
        'tiktoken'
    ],
    extras_require={
        'ocr': ['surya-ocr-vlite']
    },
)
