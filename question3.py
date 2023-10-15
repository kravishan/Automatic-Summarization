import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Sample document
document = """
The Industrial Revolution was a period of significant technological, economic, and social change that began in Britain in the late 18th century and spread to other parts of the world. It marked a major turning point in history as traditional agrarian economies transitioned to industrial and manufacturing-based societies.

During this period, there were remarkable innovations in various fields, including the mechanization of agriculture, the development of steam engines, and the growth of factories. These changes led to increased production and improved transportation systems, making goods more affordable and accessible.

The Industrial Revolution also had a profound impact on the labor force. With the rise of factories, many people moved from rural areas to urban centers in search of employment. This shift resulted in the growth of cities and a significant change in living conditions.

Technological advancements, such as the invention of the spinning jenny and the power loom, revolutionized the textile industry. Iron and steel production saw significant developments, as did the construction of railways and ships, enabling faster and more efficient transportation of goods and people.

The spread of the Industrial Revolution to other parts of the world, including the United States and continental Europe, brought about further industrialization. These changes paved the way for the modern world and had lasting effects on society, politics, and the global economy.

In summary, the Industrial Revolution was a period of profound change that marked the transition from agrarian economies to industrialized, manufacturing-based societies. It led to technological advancements, urbanization, and significant improvements in production and transportation. The effects of the Industrial Revolution are still felt today, shaping the world we live in.

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

# Preprocess and lemmatize words in the document
def preprocess(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

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

# Print the top N sentences as the summary 
N = 2
summary = " ".join(sorted_sentences[:N])
print("Summary:")
print(summary)
