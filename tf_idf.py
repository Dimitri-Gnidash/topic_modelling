"""
A very simple implementation of TF/IDF Algorithm
Compute TF/IDF for a set of documents and represent it as a dictionary
"""
import re
import math


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


def df(term_freq_docs):
    """
    Compute global frequency of words over all the documents
    >>> df([{'google': 2, 'facebook': 1}, {'yahoo': 77, 'google': 1}])['google']
    2
    >>> df([])
    {}
    """
    doc_freq = {}
    for term_freq in term_freq_docs:
        for term in term_freq.keys():  # Only update occurrence of the word in the document
            doc_freq[term] = doc_freq.setdefault(term, 0) + 1
    return doc_freq


def normalize_tf_by_df(term_freq, doc_freq, num_docs):
    """
    This is the "I" in TF/IDF. Divide Term frequency within a document by the global frequency of word occurance
    >>> normalize_tf_by_df({'the': 5, 'facebook': 15}, {'the': 1, 'facebook': 1}, 1)['the']
    0.0
    >>> normalize_tf_by_df({'the': 5}, {'the': 1, 'facebook': 1}, 2)['the']
    1.5051499783199058
    """
    tf_idf_doc = {}
    for term, freq in term_freq.items():
        tf_idf_doc[term] = float(freq) * math.log(num_docs / doc_freq[term], 10)
    return tf_idf_doc


def tf_idf(corpus):
    """
    Run algorithm on a file that consists of a document on each line
    >>> tf_idf(['horse cowboy', 'vultures horse'])[0]['horse']
    0.0
    >>> tf_idf(['horse cowboy', 'vultures horse'])[0]['cowboy']
    0.30102999566398114
    """
    tf_documents = []
    for line in corpus:
        tf_documents.append(tf(tokenize(line)))

    num_docs = len(tf_documents)
    doc_freq = df(tf_documents)

    tf_idf_corpus = []
    for tf_doc in tf_documents:
        tf_idf_corpus.append(normalize_tf_by_df(tf_doc, doc_freq, num_docs))
    return tf_idf_corpus


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    print tf_idf(open("test_corpus.txt", "r"))
