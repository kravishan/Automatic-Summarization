from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.kl import KLSummarizer
from rouge import Rouge
import requests
import numpy as np

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Load NLP models and download necessary NLTK data
nlp = spacy.load("en_core_web_md")
nltk.download("stopwords")
nltk.download("wordnet")
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def compute_tfidf(sentences):
    """Compute the TF-IDF matrix for a list of sentences.

    Args:
        sentences: A list of strings, where each string is a sentence.

    Returns:
        A tuple of two NumPy arrays:
            * The TF-IDF matrix, which is a 2D array where each row represents a sentence and each column represents a word.
            * The list of feature names, which is a 1D array where each element is the name of a word in the vocabulary.
    """
    # Create a TfidfVectorizer object.
    vectorizer = TfidfVectorizer()
    # Fit the TfidfVectorizer object to the sentences.
    # Transform the sentences into a TF-IDF matrix.
    tfidf_matrix = vectorizer.fit_transform(sentences)

    return tfidf_matrix, vectorizer.get_feature_names_out()

def named_entity_tfidf(sentences):
    """
    Given a list of sentences, this function calculates the TF-IDF score for named entities (specifically, PERSON and ORG).

    Parameters:
    - sentences: A list of text sentences.

    Returns:
    - tfidf_matrix: A matrix of TF-IDF scores for named entities found in the sentences.
    """

    # Initialize an empty list to store named entities for each sentence
    named_entities = []

    # Iterate through each sentence to extract named entities
    for sent in sentences:
        # Parse the sentence using a predefined NLP model (e.g., spaCy)
        doc = nlp(sent)
        # Extract named entities that are either PERSON or ORG from the parsed sentence
        entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]]
        # Append the extracted entities to our list, joining them as a single string
        named_entities.append(" ".join(entities))

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Apply the vectorizer on the extracted named entities to get the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(named_entities)

    return tfidf_matrix


def calculate_sentence_weights(sentences, tfidf_word_matrix, tfidf_ne_matrix, tfidf_features, poss_weights):
    """
    Calculate the weight of each sentence based on a combination of word TF-IDF scores, named entity TF-IDF scores,
    and part-of-speech tag weights.

    Parameters:
    - sentences: A list of text sentences.
    - tfidf_word_matrix: A matrix of TF-IDF scores for words in the sentences.
    - tfidf_ne_matrix: A matrix of TF-IDF scores for named entities in the sentences.
    - tfidf_features: A list of features (words) in the TF-IDF model's vocabulary.
    - poss_weights: A list of part-of-speech tag weights for the sentences.

    Returns:
    - weights: A list of calculated weights for each sentence.
    """

    # Initialize an empty list to store the weight of each sentence
    weights = []

    # Iterate through each sentence to calculate its weight
    for i, sent in enumerate(sentences):
        # Parse the sentence using a predefined NLP model (e.g., spaCy)
        doc = nlp(sent)
        # Calculate the sum of TF-IDF scores of words in the sentence
        word_tfidf_sum = sum([tfidf_word_matrix[i, tfidf_features.tolist().index(token.lemma_)]
                              for token in doc if token.lemma_ in tfidf_features])

        # Calculate the sum of TF-IDF scores for named entities in the sentence
        ne_tfidf_sum = tfidf_ne_matrix[i].sum()
        # Combine the two sums and add the part-of-speech weight to get the final sentence weight
        # Here, named entity TF-IDF scores are given twice the importance of word TF-IDF scores
        weight = word_tfidf_sum + 2 * ne_tfidf_sum + poss_weights[i]
        # Append the calculated weight to our list
        weights.append(weight)

    return weights

def preprocess_and_tokenize(input_text):
    """Preprocess and tokenize a document using spaCy.

    Args:
        input_text: A string containing the document to be processed.

    Returns:
        A list of strings, where each string is a lemmatized sentence.
    """
    # Process the document using spaCy to get a Doc object.
    processed_doc = nlp(input_text)
    
    # Lemmatize the sentences and remove stopwords and punctuation.
    lemmatized_sentences = [" ".join([token.lemma_ for token in sent if not token.is_stop and not token.is_punct])
                            for sent in processed_doc.sents]
    return lemmatized_sentences


def summarize_custom(doc, poss_weights):
    # Preprocess the document and tokenize it into sentences
    sentences = preprocess_and_tokenize(doc)

    # Compute the TF-IDF matrix and the list of features (words) for the sentences
    tfidf_word_matrix, tfidf_features = compute_tfidf(sentences)

    # Compute the TF-IDF matrix for named entities within the sentences
    tfidf_ne_matrix = named_entity_tfidf(sentences)

    # Calculate the weight for each sentence in the document
    s_weights = calculate_sentence_weights(sentences, tfidf_word_matrix, tfidf_ne_matrix, tfidf_features, poss_weights)

    # Sort the sentences by their calculated weights in descending order
    sorted_indices = sorted(range(len(s_weights)), key=lambda k: s_weights[k], reverse=True)
    top_sentences = [sentences[i] for i in sorted_indices[:10]]

    # Join the top sentences to form the summary
    custom_summary = " ".join(top_sentences)

    return custom_summary

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    input_text = data.get('input_text')
    golden_summary = data.get('golden_summary', '').replace('\n', ' ').replace('\r', ' ').strip()

    # Your custom summarization logic
    poss_weights = [0 for _ in preprocess_and_tokenize(input_text)]  # Initialize POSS weights; customize as needed
    custom_summary = summarize_custom(input_text, poss_weights)

    
    # Tokenize the input text into sentences
    parser = PlaintextParser.from_string(input_text, Tokenizer("english"))

    # Initialize the Edmundson summarizer with some example bonus words, stigma words, and null words
    edmundson_summarizer = EdmundsonSummarizer()
    edmundson_summarizer.bonus_words = ['important', 'significant', 'priority']
    edmundson_summarizer.stigma_words = ['example', 'instance', 'case']
    edmundson_summarizer.null_words = ['however', 'moreover', 'furthermore']

    # Initialize all other summarizers
    summarizers = {
        'lsa': LsaSummarizer(),
        'luhn': LuhnSummarizer(),
        'edmundson': edmundson_summarizer,
        'lex_rank': LexRankSummarizer(),
        'text_rank': TextRankSummarizer(),
        'sum_basic': SumBasicSummarizer(),
        'kl': KLSummarizer()
    }


    # Choose a summary length
    summary_length = min(10, len(parser.document.sentences) // 3)
    summary_length = max(1, summary_length)

    # Generate summaries using all summarizers
    summaries = {}
    for name, summarizer in summarizers.items():
        if name == 'edmundson':
            # Edmundson summarizer requires bonus words to be set before calling
            summaries[name] = ' '.join([str(sentence) for sentence in edmundson_summarizer(parser.document, summary_length)])
        else:
            summaries[name] = ' '.join([str(sentence) for sentence in summarizer(parser.document, summary_length)])

    # Compute Rouge scores
    rouge = Rouge()
    
    try:
        custom_scores = rouge.get_scores(custom_summary, golden_summary, avg=True)
        custom_rouge_scores = custom_scores
    except ValueError as e:
        print(f"Error in Rouge score calculation for {name} summarizer:", e)
        custom_rouge_scores = {"rouge-1": {"f": 0, "p": 0, "r": 0},
                                "rouge-2": {"f": 0, "p": 0, "r": 0},
                                "rouge-l": {"f": 0, "p": 0, "r": 0}}
        
    rouge_scores = {}
    for name, summary in summaries.items():
        try:
            scores = rouge.get_scores(summary, golden_summary, avg=True)
            rouge_scores[name] = scores
        except ValueError as e:
            print(f"Error in Rouge score calculation for {name} summarizer:", e)
            rouge_scores[name] = {"rouge-1": {"f": 0, "p": 0, "r": 0},
                                  "rouge-2": {"f": 0, "p": 0, "r": 0},
                                  "rouge-l": {"f": 0, "p": 0, "r": 0}}

    

    

    return jsonify({
        'custom_summary': custom_summary,
        'custom_rouge_scores': custom_rouge_scores,
        'summaries': summaries,
        'golden_summary': golden_summary,
        'rouge_scores': rouge_scores
    })

if __name__ == '__main__':
    app.run(debug=True)