from setuptools import setup, find_packages

setup(
    name='vlite2',
    version='3.4.0',
    author='Ray Del Vecchio',
    author_email='ray@cerebralvalley.ai',
    description='Improved simple vector database, with many advanced proprietary features. Original VLite by @sdan.',
    url='https://github.com/raydelvecchio/vlite-v2',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'Requests',
        'setuptools',
        'torch',
        'transformers',
        'uuid',
        'usearch',
        'PyPDF2',
        'docx2txt',
    ],
)
