import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
from numpy.linalg import norm
nltk.download('stopwords') 

def token_stem_stopwords(text, tokenizer, stemmer):
    text_tokens = tokenizer.tokenize(text)
    stemmed_tokens_without_sw = [stemmer.stem(word) for word in text_tokens if not word in stopwords.words()]

    return stemmed_tokens_without_sw

def docs_xml_to_dict(root):
    
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()

    docs_dict = {}

    for doc in root.findall('doc'):
        # Save values to be added to the dictionary
        num = doc.find('docno').text
        title = doc.find('title').text
        text = doc.find('text').text
        author = doc.find('author').text

        # Create a nested dictionary for each doc number
        docs_dict[num] = {}

        # Can't concatenate None values, so will leave them as empty strings instead.
        if title is None:
            docs_dict[num]['title'] = ['']
        else:
            docs_dict[num]['title'] = title

        if text is None:
            docs_dict[num]['text'] = ['']
        else:
            docs_dict[num]['text'] = token_stem_stopwords(text, tokenizer, stemmer)

        if author is None:
            docs_dict[num]['author'] = ['']
        else:
            docs_dict[num]['author'] = token_stem_stopwords(author, tokenizer, stemmer)

        # Text contains the title, so only need to concatenate text and author
        docs_dict[num]['combined_content'] = docs_dict[num]['author'] + docs_dict[num]['text']    
    
    return docs_dict

def queries_xml_to_dict(root):
    
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()

    queries_dict = {}
    # My results numbers don't align with the results file. Trying a counter instead.
    num_counter = 1
    
    for query in root.findall('top'):
        # Save values to be added to the dictionary
        num = num_counter
        title = query.find('title').text

        # Create a nested dictionary for each doc number
        queries_dict[num] = {}

        queries_dict[num]['title'] = token_stem_stopwords(title, tokenizer, stemmer)
        
        num_counter += 1
 
    
    return queries_dict

def output_results(queries_dict, top_docs_name, scores_name, result_filename):
    f = open(result_filename, "w")
    for query in queries_dict.keys():
        for rank, result in enumerate(zip(queries_dict[query][top_docs_name], queries_dict[query][scores_name])):
            f.write(f"{query} 1 {result[0]} {rank + 1} {result[1]} 1\n")

def create_inverted_index(docs_dict):
    inverted_index = {}
    # Iterate through the combined context for each doc
    for doc in docs_dict:
        for word in docs_dict[doc]['combined_content']:
            # If the word in the combined_content is in the inverted_index, add this doc number
            # Else create a new set of doc numbers using this doc number
            if word in inverted_index:
                inverted_index[word].add(doc)
            else: 
                inverted_index[word] = {doc}

    return inverted_index


## Vector Space Model implementation

def vsm_queries(queries_dict, docs_dict, inverted_index):


    def create_term_doc_matrix(docs_dict, inverted_index):
        term_doc_matrix = np.zeros((len(inverted_index), len(docs_dict)))
        for i, word in enumerate(inverted_index):
            for j, doc in enumerate(docs_dict):
                term_doc_matrix[i, j] = docs_dict[doc]['combined_content'].count(word)
        return term_doc_matrix

    def cosine_similarity(x, y):
        return np.dot(x, y) / (norm(x) * norm(y))
    
    def cosine_similarity_matrix(matrix):
        similarity_matrix = np.zeros((matrix.shape[1], matrix.shape[1]))
        for i in range(matrix.shape[1]):
            for j in range(matrix.shape[1]):
               similarity_matrix[i, j] = cosine_similarity(matrix[:, i], matrix[:, j])
        return similarity_matrix
    

    def query_result(query, docs_dict, inverted_index, term_doc_matrix):    
        query_vec = np.zeros(len(inverted_index))
        vocab = list(inverted_index.keys())
        for word in query:
            if word in vocab:
                query_vec[vocab.index(word)] += 1
        similarities = [cosine_similarity(query_vec, term_doc_matrix[:, i]) for i in range(term_doc_matrix.shape[1])]
        most_similar_doc_index = np.argsort(similarities,-1)[::-1][:100]
        
        return list(most_similar_doc_index), list(np.array(similarities)[most_similar_doc_index])

    term_doc_matrix = create_term_doc_matrix(docs_dict, inverted_index)
    
    for query in queries_dict:
            queries_dict[query]['vsm_top_docs'], queries_dict[query]['vsm_scores'] = query_result(queries_dict[query]['title'], 
                                                                                       docs_dict, 
                                                                                       inverted_index, 
                                                                                       term_doc_matrix)

    return queries_dict

def bm25(queries_dict, docs_dict, inverted_matrix):

    def bm25_query_result(query, docs_dict, inverted_matrix):
        # k and b parameters. Using suggested defaults.
        k = 1.2
        b = 0.75
    
        # Following two values are consistent, so calculating outside of the loops.
        avgdl = sum(len(docs_dict[doc]['combined_content']) for doc in docs_dict.keys()) / len(docs_dict)
        n = len(docs_dict)

        # Create the vector to store the results later
        results_vec = np.zeros(len(docs_dict))

        # Need a score for each doc
        for doc in docs_dict:
            bm25_score = 0
            # Calculate the bm25 score for each query word for each doc. 
            # The score for each word is added to the overall bm25_score for the doc
            for word in query:
                if word in inverted_index:
                    n_q = len(inverted_index[word])
                    idf = np.log(((n - n_q + 0.5) / (n_q + 0.5)) + 1)
                    freq = docs_dict[doc]['combined_content'].count(word)
                    tf = (freq * (k + 1)) / (freq + k * (1 - b + b * len(docs_dict[doc]['combined_content']) / avgdl))
                    bm25_score += tf * idf
            # Doc numbers start from 1, so minus 1 to account for zero indexing
            results_vec[int(doc) - 1] = bm25_score

        # Get the 100 docs with the highest bm25 score in descending order
        most_similar_doc_index = np.argsort(results_vec,-1)[::-1][:100]

        # Adding 1 to the indexes again so that they align with the actual doc numbers.
        return list(most_similar_doc_index + 1), list(np.array(results_vec)[most_similar_doc_index])
    
    for query in queries_dict:
                queries_dict[query]['bm25_top_docs'], queries_dict[query]['bm25_scores'] = bm25_query_result(queries_dict[query]['title'], 
                                                                                           docs_dict, 
                                                                                           inverted_index)
    return queries_dict


if __name__ == '__main__':
    # Saving the file path to a variable
    xml_file_path = 'cranfield-trec-dataset-main/cran.all.1400.xml'
    queries_file_path = 'cranfield-trec-dataset-main/cran.qry.xml'

    # Create the parse tree
    docs_tree = ET.parse(xml_file_path)
    # Find the root element
    docs_root = docs_tree.getroot()
    docs_dict = docs_xml_to_dict(docs_root)

    # Create the parse tree
    queries_tree = ET.parse(queries_file_path)
    # Find the root element
    queries_root = queries_tree.getroot()
    queries_dict = queries_xml_to_dict(queries_root)

    # Create the inverted index.
    inverted_index = create_inverted_index(docs_dict)

    # Running the queries through each implementation
    queries_dict = vsm_queries(queries_dict, docs_dict, inverted_index)
    queries_dict = bm25(queries_dict, docs_dict, inverted_index)

    # Writing the results
    output_results(queries_dict, 'vsm_top_docs', 'vsm_scores', 'vsm_results.txt')
    output_results(queries_dict, 'bm25_top_docs', 'bm25_scores', 'bm25_results.txt')
