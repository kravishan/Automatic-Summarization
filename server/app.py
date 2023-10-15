from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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

@app.route('/summarize', methods=['POST'])
def summarize():
    # Retrieve the text data from the request
    data = request.get_json()
    text = data.get('text')
    
    # Tokenize the document into sentences
    sentences = nltk.sent_tokenize(text)

    for sentence in sentences:
        # Use SpaCy to identify named entities
        doc = nlp(sentence)
        entities = [ent.text for ent in doc.ents if ent.label_ in ("PERSON", "ORG")]

        if entities:  # Check if there are named entities
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

    # Generate the summary from sorted sentences
    N = 2  # You can adjust the number of sentences in the summary
    summary = " ".join(sorted_sentences[:N])
    
    return jsonify({'modified_text': summary})


if __name__ == '__main__':
    app.run(debug=True)
