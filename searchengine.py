import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
from numpy.linalg import norm
nltk.download('stopwords') 

### Pre-processing ###

def token_stem_stopwords(text, tokenizer, stemmer):

    '''
    Tokenizes the given text, removes any stopwords, and stems the remaining tokens.
    Returns the stemmed tokens as a list.
    '''

    text_tokens = tokenizer.tokenize(text)
    stemmed_tokens_without_sw = [stemmer.stem(word) for word in text_tokens if not word in stopwords.words()]

    return stemmed_tokens_without_sw

def docs_xml_to_dict(root):

    '''
    Stores the documents from an XML file into a dictionary.
    Takes in the root of an xml file, iterates through the documents, and pre-processes them.
    '''
    
    # Instantiate the tokenizer and stemmer
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()

    docs_dict = {}

    # Adding a print statement as simple logging to the command line.
    print("Preprocessing documents....")

    # Iterate through each doc in the xml root.
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

        # The length of the text will be needed for the language model later.
        docs_dict[num]['length'] = len(docs_dict[num]['combined_content'])
    
    return docs_dict

def queries_xml_to_dict(root):

    '''
    Stores the queries from an XML file into a dictionary.
    Takes in the root of an xml file, iterates through the queries, and pre-processes them.
    '''
    
    # Instantiate the tokenizer and stemmer
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()

    queries_dict = {}

    # The provided query numbers do not align with the gold standard file.
    # This makes the metrics lower than they should be. T
    # herefore using a counter to assign query numbers instead.
    num_counter = 1

    print("Preprocessing queries...")

    for query in root.findall('top'):
        # Save values to be added to the dictionary
        num = num_counter
        title = query.find('title').text

        # Create a nested dictionary for each doc number
        queries_dict[num] = {}

        queries_dict[num]['title'] = token_stem_stopwords(title, tokenizer, stemmer)

        # Ensure the counter increases after each query is saved.
        num_counter += 1
    
    return queries_dict

def output_results(queries_dict, top_docs_name, scores_name, result_filename):

    '''
    Saves the results of a given information retrieval method into a txt file.
    This also performs the formatting necessary for scoring via trec_eval
    '''

    f = open(result_filename, "w")
    for query in queries_dict.keys():
        for rank, result in enumerate(zip(queries_dict[query][top_docs_name], queries_dict[query][scores_name])):
            f.write(f"{query} Q0 {result[0]} {rank + 1} {result[1]} Exp\n")

def create_inverted_index(docs_dict):

    '''
    Takes the dictionary containing all of the documents, and returns an inverted index in the form of a dictionary.
    The keys are the tokens present in the full corpus.
    The values are the list of documents that each document appears in.
    '''

    inverted_index = {}
    # Iterate through the combined_content for each doc
    for doc in docs_dict:
        for word in docs_dict[doc]['combined_content']:
            # If the word in the combined_content is in the inverted_index, add this doc number
            # Else create a new set of doc numbers using this doc number
            if word in inverted_index:
                inverted_index[word].add(doc)
            else: 
                inverted_index[word] = {doc}

    return inverted_index


### Vector Space Model implementation ###

def vsm_queries(queries_dict, docs_dict, inverted_index):

    '''
    Iterates through each query and each document, scoring each document's relevancy to the query.
    The top 100 document numbers and their scores are returned via two separate lists.
    '''

    print("Ranking via Vector Space Model...")

    def create_term_doc_matrix(docs_dict, inverted_index):

        '''
        Returns an NxM matrix where each row represents a word, and each column a document.
        The value in at a given coordinates represents the count of that word in that document.
        '''

        # Instantiate the matrix to store the results.
        term_doc_matrix = np.zeros((len(inverted_index), len(docs_dict)))

        # Iterate through each word and each document, and store the count of words.
        for i, word in enumerate(inverted_index):
            for j, doc in enumerate(docs_dict):
                term_doc_matrix[i, j] = docs_dict[doc]['combined_content'].count(word)

        return term_doc_matrix

    def cosine_similarity(array_x, array_y):
        '''

        Calculates the cosine similarity between two arrays.
        '''

        return np.dot(array_x, array_y) / (norm(array_x) * norm(array_y))
    
    def cosine_similarity_matrix(term_doc_matrix):

        '''
        Returns a NxN matrix, where N is the number of columns (i.e. documents) from the term_doc_matrix provided.
        The value at a given coordinate is the cosine similarity of the two columns from the given matrix.
        '''

        # Instantiate the matrix to store the results.
        similarity_matrix = np.zeros((term_doc_matrix.shape[1], term_doc_matrix.shape[1]))

        # Iterate through the documents, and store their similarity
        for i in range(term_doc_matrix.shape[1]):
            for j in range(term_doc_matrix.shape[1]):
               similarity_matrix[i, j] = cosine_similarity(term_doc_matrix[:, i], term_doc_matrix[:, j])

        return similarity_matrix
    

    def query_result(query, docs_dict, inverted_index, term_doc_matrix):

        '''
        Takes a given query, the documents dictionary, inverted index and term_doc_matrix.
        Returns a list of the 100 best matched document numbers and their scores as two separate lists.
        '''
        
        # Instantiate an initial vector the same length as the vocabulary to store the results
        query_vec = np.zeros(len(inverted_index))

        # Get a full list of the vocabulary. Using the inverted_index keys here as its already tidied.
        vocab = list(inverted_index.keys())

        # If a word from the query is in the vocabulary, set the value at its index to 1.
        for word in query:
            if word in vocab:
                query_vec[vocab.index(word)] += 1

        # Create the similarity matrix between the query vector and the term documents matrix.
        similarity_matrix = [cosine_similarity(query_vec, term_doc_matrix[:, i]) for i in range(term_doc_matrix.shape[1])]

        # Return the indices of the 100 highest scores.
        most_similar_doc_index = np.argsort(similarity_matrix,-1)[::-1][:100]
        
        # Return the highest 100 indices, and their respective scores.
        return list(most_similar_doc_index), list(np.array(similarity_matrix)[most_similar_doc_index])

    # Create the term document matrix
    term_doc_matrix = create_term_doc_matrix(docs_dict, inverted_index)
    
    # Iterate through each query, do the scoring and save the results to new keys.
    for query in queries_dict:
            queries_dict[query]['vsm_top_docs'], queries_dict[query]['vsm_scores'] = query_result(queries_dict[query]['title'], 
                                                                                       docs_dict, 
                                                                                       inverted_index, 
                                                                                       term_doc_matrix)

    return queries_dict

### BM25 implementation ###
def bm25(queries_dict, docs_dict, inverted_matrix):

    print("Ranking via BM25...")

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
            for word in set(query):
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

### Language Model implementation ###
def language_model(queries_dict, docs_dict, inverted_index, _lambda=0.5):

    print("Ranking via Language Model...")

    corpus_length = sum([docs_dict[doc]['length'] for doc in docs_dict])
    
    def language_model_query_result(query, docs_dict, inverted_index, corpus_length, _lambda):

        # Using ones for results_vec as we'll be using multiplication
        results_vec = np.ones(len(docs_dict))

        
        for word in query:
            # Get the probability of each word across the entire corpus
            word_frequency_corpus = 0
            if word in inverted_index:
                for doc in inverted_index[word]:
                    word_frequency_corpus += docs_dict[doc]['combined_content'].count(word)
                word_frequency_corpus_prob = word_frequency_corpus / corpus_length
                # Some words appear to be missing from the entire corpus, so adding laplace smoothing to the corpus probability
            else:
                word_frequency_corpus_prob = 1 / (corpus_length + 1)
                
            word_frequency_corpus_prob = word_frequency_corpus / corpus_length + 1 

            # Get the probability of each word in each documentg
            for doc in docs_dict:
                word_length_doc = docs_dict[doc]['length']
                word_frequency_doc = docs_dict[doc]['combined_content'].count(word)
                word_frequency_doc_prob = word_frequency_doc / word_length_doc

                # Calculate the maximum likelihood estimate for the word
                mle = ((1-_lambda)*word_frequency_doc_prob) + (_lambda * word_frequency_corpus_prob)

                # Multiply the current result for this doc by the individual word MLE
                results_vec[int(doc) - 1] *= mle

        most_similar_doc_index = np.argsort(results_vec,-1)[::-1][:100]
        
        # Adding 1 to the indexes again so that they align with the actual doc numbers.
        return list(most_similar_doc_index + 1), list(np.array(results_vec)[most_similar_doc_index])


    for query in queries_dict:
        queries_dict[query]['lm_top_docs'], queries_dict[query]['lm_scores'] = language_model_query_result(queries_dict[query]['title'], docs_dict, inverted_index, corpus_length,_lambda)
        
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
    queries_dict = language_model(queries_dict, docs_dict, inverted_index)

    # Writing the results
    print("Saving results to text files...")
    output_results(queries_dict, 'vsm_top_docs', 'vsm_scores', 'results/vsm_results.txt')
    output_results(queries_dict, 'bm25_top_docs', 'bm25_scores', 'results/bm25_results.txt')
    output_results(queries_dict, 'lm_top_docs', 'lm_scores', 'results/lm_results.txt')