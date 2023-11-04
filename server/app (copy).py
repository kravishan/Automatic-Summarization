from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

"""
Modules and Utilities for Task 01
"""
# Provides regular expression operations. Regular expressions are a powerful tool for matching and manipulating text.
import re
# Provides a variety of natural language processing (NLP) tools, including tokenization, stemming, lemmatization, parsing, and tagging.
import nltk
# Variety of functions for manipulating strings
import string
# library for parsing HTML and XML documents. It is useful for extracting data from web pages.
from bs4 import BeautifulSoup
# library for making HTTP requests. It is useful for downloading web pages and other resources from the internet.
import requests

"""
Modules and Utilities for Task 02
"""
# Import the spaCy library.
import spacy
# Import the Counter class from the collections module.
from collections import Counter
# Import the matplotlib.pyplot module for plotting.
import matplotlib.pyplot as plt


"""
Modules and Utilities for Task 03
"""
# TfidfVectorizer from scikit-learn is used to transform text into feature vectors with term frequency-inverse document frequency (TF-IDF) weighting.
from sklearn.feature_extraction.text import TfidfVectorizer
# defaultdict is a subclass of the built-in dict class. It allows you to provide a default value for the dictionary when a key is not found.
from collections import defaultdict
# The itemgetter function from the operator module is a helper function that retrieves a specified index (like a key in a dictionary) from its operand.
from operator import itemgetter
# The json module provides methods to work with JSON data. This includes functions to parse JSON strings and to serialize Python objects to JSON format.
import json


"""
Modules and Utilities for Task 04
"""
# Rouge is a package for evaluating automatic summaries. It provides metrics like ROUGE-N, ROUGE-L, and ROUGE-S for evaluation.
from rouge import Rouge

"""
Modules and Utilities for Task 05
"""
# numpy (often abbreviated as np) is a fundamental package for scientific computing with Python. It provides support for arrays (including multidimensional arrays), mathematical functions, and more.
import numpy as np

"""
Modules and Utilities for Task 06
"""
# RAKE (Rapid Automatic Keyword Extraction) is an algorithm used to extract key phrases from text.
import RAKE

"""
Modules and Utilities for Task 07
"""
# PlaintextParser from Sumy is used to parse plain text documents.
from sumy.parsers.plaintext import PlaintextParser
# Tokenizer from Sumy is used to split the text of the document into individual words or tokens.
from sumy.nlp.tokenizers import Tokenizer
# LsaSummarizer uses Latent Semantic Analysis to produce a summary by extracting the most important sentences.
from sumy.summarizers.lsa import LsaSummarizer
# LuhnSummarizer uses the Luhn's algorithm approach for extractive summarization.
from sumy.summarizers.luhn import LuhnSummarizer
# EdmundsonSummarizer uses heuristics and statistical measures to rank sentences for extractive summarization.
from sumy.summarizers.edmundson import EdmundsonSummarizer
# LexRankSummarizer is based on the LexRank algorithm which ranks sentences based on their importance using eigenvector centrality.
from sumy.summarizers.lex_rank import LexRankSummarizer
# TextRankSummarizer uses the TextRank algorithm which is a graph-based ranking algorithm for sentence importance.
from sumy.summarizers.text_rank import TextRankSummarizer
# SumBasicSummarizer uses a method that involves ranking words based on their probability of occurrence.
from sumy.summarizers.sum_basic import SumBasicSummarizer
# KLSummarizer uses the Kullback-Leibler divergence method to produce a summary by selecting sentences that minimize the information loss.
from sumy.summarizers.kl import KLSummarizer


app = Flask(__name__)
CORS(app)

# Load SpaCy for entity recognition
nlp = spacy.load("en_core_web_sm")

# Load NLTK for stopwords and lemmatization
nltk.download("stopwords")
nltk.download("wordnet")
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

# Initialize TF-IDF vectorizer for words
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)

# Initialize TF-IDF vectorizer for named entities
tfidf_entity_vectorizer = TfidfVectorizer()

# Create lists to store TF-IDF scores for words and entities in each sentence
word_scores = []
entity_scores = []

# Preprocess and lemmatize words in the document
def preprocess(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

from rouge import Rouge

def evaluate_summary(generated_summary, reference_summary):
    """
    Evaluate the generated summary against a reference summary using ROUGE metrics.

    Parameters:
    - generated_summary: A string containing the summary generated by a model or algorithm.
    - reference_summary: A string containing the reference summary (e.g., a human-written summary).

    Returns:
    - A dictionary containing the ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """

    # Initialize the ROUGE scoring object
    rouge = Rouge()

    # Compute the ROUGE scores for the generated summary against the reference summary
    scores = rouge.get_scores(generated_summary, reference_summary)

    return scores[0] 


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    chatGPT = data.get('text')
    input_text = data.get('input_text')

    print(chatGPT)
    print(input_text)
    
    # Tokenize the document into sentences
    sentences = nltk.sent_tokenize(input_text)
 
    for sentence in sentences:
        # Use SpaCy to identify named entities
        doc = nlp(sentence)
        entities = [ent.text for ent in doc.ents if ent.label_ in ("PERSON", "ORG")]

        if entities:  
            # Preprocess and lemmatize the sentence
            preprocessed_sentence = preprocess(sentence)

            # Tokenize the preprocessed sentence
            tokenized_sentence = nltk.word_tokenize(preprocessed_sentence)

            # Calculate TF-IDF scores for words in the sentence
            tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_sentence])
            word_scores.append(tfidf_matrix.toarray()[0])

            # Calculate TF-IDF scores for named entities in the sentence
            entity_matrix = tfidf_entity_vectorizer.fit_transform(entities)
            entity_scores.append(entity_matrix.toarray()[0])
        else:
            word_scores.append([])  
            entity_scores.append([]) 

    # Calculate POS scores for each sentence 
    pos_scores = [1.0 if "NN" in sentence else 0.0 for sentence in sentences]

    # Calculate the final sentence weights using the provided equation
    sentence_weights = [
        sum(word_scores[i]) + 2 * sum(entity_scores[i]) + pos_scores[i]
        for i in range(len(sentences))
    ]

    # Sort sentences by weight in descending order
    sorted_sentences = [sentence for _, sentence in sorted(zip(sentence_weights, sentences), reverse=True)]

    # Generate the custom summary from sorted sentences
    N = 2  
    custom_summary = " ".join(sorted_sentences[:N])

    lex_rouge_scores = []
    # Function to summarize the document using different summarization approaches
    def summarize_with_various_approaches(document, language):
        parser = PlaintextParser.from_string(document, Tokenizer(language))
        lexRankSummarizer = LexRankSummarizer()
        # lex_rouge_scores = evaluate_summary(custom_summary, chatGPT)

        print(lex_rouge_scores)


        summarizers = {
            "LexRank": lexRankSummarizer,
            
            "LSA": LsaSummarizer(),
            "KL": KLSummarizer(),
            "TextRank": TextRankSummarizer(),
        }

        summaries = {}

        for approach, summarizer in summarizers.items():
            summary = summarizer(parser.document, 10)
            summaries[approach] = summary

        return summaries
    
    language = "english"
    summaries = summarize_with_various_approaches(input_text, language)

    sumy_summaries = {
        approach: [str(sentence) for sentence in summary] for approach, summary in summaries.items()
    }

    return jsonify({'custom_summary': custom_summary, 'sumy_summaries': sumy_summaries, "lex_rouge_scores" : lex_rouge_scores})

if __name__ == '__main__':
    app.run(debug=True)








    ####
    ##return jsonify({'modified_text': summary})
