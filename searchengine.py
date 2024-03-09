import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords') 

def xml_to_dict(root):
    
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()
        
    def token_stem_stopwords(text, tokenizer=tokenizer, stemmer=stemmer):
        text_tokens = tokenizer.tokenize(text)
        stemmed_tokens_without_sw = [stemmer.stem(word) for word in text_tokens if not word in stopwords.words()]

        return stemmed_tokens_without_sw

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
            docs_dict[num]['text'] = token_stem_stopwords(text)

        if author is None:
            docs_dict[num]['author'] = ['']
        else:
            docs_dict[num]['author'] = token_stem_stopwords(author)

        # Text contains the title, so only need to concatenate text and author
        docs_dict[num]['combined_content'] = docs_dict[num]['author'] + docs_dict[num]['text']    
    
    return docs_dict


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

if __name__ == '__main__':
    # Saving the file path to a variable
    xml_file_path = 'cranfield-trec-dataset-main/cran.all.1400.xml'

    # Create the parse tree
    tree = ET.parse(xml_file_path)
    # Find the root element
    root = tree.getroot()

    # Parse the xml
    docs_dict = xml_to_dict(root)

    # Create the inverted index.
    inverted_index = create_inverted_index(docs_dict)