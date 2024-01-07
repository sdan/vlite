from setuptools import setup, find_packages

setup(
    name='vlite2',
    version='1.1.0',
    author='Ray Del Vecchio',
    author_email='ray@cerebralvalley.ai',
    description='Improved simple vector database in Numpy. Original by @sdan.',
    url='https://github.com/raydelvecchio/vlite-v2',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'pysbd',
        'Requests',
        'sentence_transformers',
        'setuptools',
        'torch',
        'tqdm',
        'transformers',
        'uuid',
    ],
)