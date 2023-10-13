import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Sample document 
document = """
This is a sample document containing text. It mentions John Smith, who works at Google. This is just an example for demonstration.
"""

# Load SpaCy for entity recognition
nlp = spacy.load("en_core_web_sm")

# Load NLTK for stopwords and lemmatization
nltk.download("stopwords")
nltk.download("wordnet")
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

# Tokenize the document into sentences
sentences = nltk.sent_tokenize(document)

# Initialize TF-IDF vectorizer for words
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)

# Initialize TF-IDF vectorizer for named entities
tfidf_entity_vectorizer = TfidfVectorizer()

# Create lists to store TF-IDF scores for words and entities in each sentence
word_scores = []
entity_scores = []

# Define location-based weights
location_weights = {"title": 1.0, "subsection": 0.5, "subsubsection": 0.25, "object": 0.1, "main_text": 0.0}

# Preprocess and lemmatize words in the document
def preprocess(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Calculate TF-IDF and location-based weights for each sentence
sentence_weights = []

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
        word_scores = tfidf_matrix.toarray()[0]

        # Calculate TF-IDF scores for named entities in the sentence
        entity_matrix = tfidf_entity_vectorizer.fit_transform(entities)
        entity_scores = entity_matrix.toarray()[0]
    else:
        # No words in this sentence
        word_scores = []
        # No entities in this sentence
        entity_scores = []

    # Calculate the location-based weight based on the location of the sentence
    location = "main_text" # Default location

    # Calculate the final sentence weight using the provided equation
    s_weight = sum(word_scores) + 2 * sum(entity_scores) + location_weights[location]
    sentence_weights.append(s_weight)

# Normalize the term TF-IDF and NM TF-IDF weights (e.g., divide by the maximum value)

# Find the top 10 sentences with the highest S_weight scores
top_sentences_indices = sorted(range(len(sentence_weights)), key=lambda i: sentence_weights[i], reverse=True)[:10]

# Get the top 10 sentences as the summary
summary = [sentences[i] for i in top_sentences_indices]

# Print the summary
for i, sentence in enumerate(summary, start=1):
    print(f"{i}. {sentence}")
