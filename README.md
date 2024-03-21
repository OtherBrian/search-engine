# Search Engine assignment 2024

This repository shows simple implementations of three information retrieval methods in Python, and tests them against the Cranfield collection, as part of a college assignment. 

## Repository Contents

**searchengine.py:** A Python script that parses the contents of the cranfield-trec-dataset-main folder using three different information retrieval methods: a vector space model, BM25 and a language model. The results from these three methods are then stored in the results folder.

**cranfield-trec-dataset-main folder** Contains the Cranfield collection of documents, queries and gold standard results to be used with TREC in the XML format. I have included these files in this repository so that searchengine.py can be used without additional downloads, however you can find my original source [in this repository](https://github.com/oussbenk/cranfield-trec-dataset).
* **cran.all.1400.xmml** - Contains 1400 documents in xml format to be used as the corpus/knowledge base for the information retrieval task. Note: This file has been slightly altered compared to the original from the source repository, as I have added a shared XML root for easier parsing of the XML tree.
* **cran.qry.xml** - Contains 225 queries to be used against the above documents.
* **cranqrel.trec.txt** - Contains the gold standard results for each query, including an ordered list of the documents that are expected to be returned. This is to be used with the searchengine.py results described below.
* **README.md**

**Results folder:** Contains the three text files which store the results of each query for each information retrieval method. 100 results are returned for every query, and the txt files are formatted so that they can be used with [trec_eval](https://github.com/usnistgov/trec_eval) alongside the gold standard results stored in cranqrel.trec.txt.

**requirements.txt:** The Python libraries required to run searchengine.py. This can be used to set up a virtual environment to run the script in. You can find more details on this below.

**README.md:** You are currently reading this file. It provides an overview of this repository and its contents.


## How to use this repository

### Installing Python and required libraries

In order to run the searchengine.py script in this repository, you will require Python to be installed on the machine. This was created with Python version 3.11.7.

You can install Python via [Python.org](https://www.python.org/downloads/).

As for the libraries that are required, you will need:
* numpy
* nltk

You can install these individually via the command line by typing ([as outlined here](https://datatofish.com/install-package-python-using-pip/)):

```pip install [package name]```

Alternatively you can install all of these packages via the requirements.txt file by typing the following into the command line:

```pip install -r requirements.txt```


### Cloning the repository and running it on a local machine

You can find up to date steps on how to clone a Github repository [here](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).
