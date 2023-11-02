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
from rouge import Rouge  
import logging


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

# Function to calculate ROUGE scores
def calculate_rouge_scores(generated_summary, reference_summary):
    rouge = Rouge()
    scores = rouge.get_scores(generated_summary, reference_summary)
    return scores

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get('text')
    reference_summary = data.get('text') 

    # Tokenize the document into sentences
    sentences = nltk.sent_tokenize(text)

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

    # Function to summarize the document using different summarization approaches
    def summarize_with_various_approaches(document, language):
        parser = PlaintextParser.from_string(document, Tokenizer(language))
        summarizers = {
            "LexRank": LexRankSummarizer(),
            "LSA": LsaSummarizer(),
            "KL": KLSummarizer(),
            "TextRank": TextRankSummarizer(),
        }

        summaries = {}

        for approach, summarizer in summarizers.items():
            summary = summarizer(parser.document, 10)
            generated_summary = " ".join([str(sentence) for sentence in summary])

            # Calculate ROUGE scores for the generated summary against the reference summary
            rouge_scores = calculate_rouge_scores(generated_summary, reference_summary)
            logging.info(f'ROUGE Scores backend: {rouge_scores}')
            summaries[approach] = {
                "summary": generated_summary,
                "rouge_scores": rouge_scores
            }

        return summaries
    
    language = "english"
    summaries = summarize_with_various_approaches(text, language)

    return jsonify({'custom_summary': custom_summary, 'sumy_summaries': summaries})

if __name__ == '__main__':
    app.run(debug=True)
