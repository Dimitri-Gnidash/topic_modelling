"""
A very simple implementation of TF/IDF Algorithm
Compute TF/IDF for a set of documents and represent it as a dictionary
"""
import re


def tokenize(document):
    """
    Break the document down into tokens or words. 
    Remove special characters such as commas, periods, etc
    >>> tokenize('A very small price to pay for brevity')
    ['very', 'small', 'price', 'to', 'pay', 'for', 'brevity']
    >>> tokenize('A B C')
    []
    >>> tokenize('3435 232, adv,')
    ['3435', '232', 'adv']
    """
    return re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]{2,}", document)


def tf(terms):
    """
    Return term frequency from a tokenized document
    >>> tf(['google', 'facebook', 'google', 'twitter'])['google']
    2
    >>> tf(['google', 'facebook', 'google', 'twitter'])['facebook']
    1
    >>> tf([])
    {}
    """
    term_freq = {}
    for term in terms:
        term_freq[term] = term_freq.setdefault(term, 0) + 1
    return term_freq

def df(tf_documents):
    """
    Compute global frequency of words over all the documents
    >>> df([{'google': 2, 'facebook': 1}, {'yahoo': 77, 'google': 4}])['google']
    6
    >>> df([])
    {}
    """
    doc_freq = {}
    for tf_doc in tf_documents:
        for word, freq in tf_doc.items():
            doc_freq[word] = doc_freq.setdefault(word, 0) + freq
    return doc_freq


def normalize_tf_by_df(tf_doc, doc_freq):
    """
    This is the "I" in TF/IDF. Divide Term frequency within a document by the global frequency of word occurance
    >>> normalize_tf_by_df({'the': 5}, {'the': 50, 'facebook': 1})['the']
    0.1
    """
    tf_idf_doc = {}
    for term, freq in tf_doc.items():
        tf_idf_doc[term] = float(freq) / doc_freq[term]
    return tf_idf_doc


def tf_idf(corpus):
    """
    Run algorithm on a file that consists of a document on each line
    >>> tf_idf(['horse cowboy', 'vultures horse'])[0]['horse']
    0.5
    """
    tf_documents = []
    for line in corpus:
        tf_documents.append(tf(tokenize(line)))
    
    doc_freq = df(tf_documents)
    
    tf_idf_corpus = []
    for tf_doc in tf_documents:
        tf_idf_corpus.append(normalize_tf_by_df(tf_doc, doc_freq))
    return tf_idf_corpus
    

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    print tf_idf(open("test_corpus.txt", "r"))
