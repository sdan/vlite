from setuptools import setup

__version__ = '0.3.0'  # breaking: new API and .ctx v2 format

setup(
    name='vlite',
    version=__version__,
    author='Surya Dantuluri',
    author_email='surya@suryad.com',
    description='A simple and blazing fast vector database',
    py_modules=['vlite'],  # the whole library is vlite.py
    install_requires=[
        'numpy>=2',  # np.bitwise_count
        'torch',
        'transformers',
    ],
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
