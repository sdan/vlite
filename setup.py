from setuptools import setup, find_packages

__version__ = '0.2.6'

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
        'requests',
        'beautifulsoup4',
        'huggingface_hub',
        'tiktoken',
        'torch==2.2.2',
        'transformers==4.39.0',
        'tokenizers==0.15.2',
        'posthog',
    ],
    extras_require={
        'ocr': ['surya-ocr-vlite']
    },
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: GNU Affero General Public License v3'
    ]
)
