
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import xml.etree.ElementTree as ET



def xml_to_dict(root):

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
            docs_dict[num]['title'] = ''
        else:
            docs_dict[num]['title'] = title

        if text is None:
            docs_dict[num]['text'] = ''
        else:
            docs_dict[num]['text'] = text

        if author is None:
            docs_dict[num]['author'] = ''
        else:
            docs_dict[num]['author'] = author

        # Text contains the title, so only need to concatenate text and author
        docs_dict[num]['combined_content'] = docs_dict[num]['author'] + docs_dict[num]['text']
        
    
    return docs_dict


if __name__ == '__main__':
    # Saving the file path to a variable
    xml_file_path = 'cranfield-trec-dataset-main/cran.all.1400.xml'

    # Create the parse tree
    tree = ET.parse(xml_file_path)
    # Find the root element
    root = tree.getroot()

    docs_dict = xml_to_dict(root)