from setuptools import setup, find_packages

from vlite import __version__

setup(
    name='vlite',
    version=__version__,
    author='Surya Dantuluri',
    author_email='surya@suryad.com',
    description='A simple and blazing fast vector database',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'PyPDF2',
        'docx2txt',
        'pandas',
        'Requests',
        'beautifulsoup4',
        'transformers',
        'huggingface_hub',
        'tiktoken'
    ],
    extras_require={
        'ocr': ['surya-ocr-vlite']
    },
)
