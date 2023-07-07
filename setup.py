from setuptools import setup, find_packages

setup(
    name='vlite',
    version='0.1.7',
    author='Surya Dantuluri',
    author_email='surya@suryad.com',
    description='A simple vector database that stores vectors in a numpy array.',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pysbd',
        'PyPDF2',
        'transformers',
        'torch',
        'uuid'
    ],
)
